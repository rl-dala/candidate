"""
Постобработка OCR паспорта РФ: принять / отклонить скан

Предсказание:
    python postprocess.py data/test_predictions.jsonl answers.jsonl
Обучение (создает model.pkl, печатает компромисс охват/ошибки и прогноз):
    python postprocess.py data/train_predictions.jsonl data/train_oof_answers.jsonl --train data/train_labels.jsonl --max-wrong 12 --repeats 3
    python evaluate.py data/train_oof_answers.jsonl data/train_labels.jsonl

Python 3.14

https://www.consultant.ru/document/cons_doc_LAW_373184/
Структура MRZ для российских паспортов:
1.  1.1-1.2:    Тип паспорт     P<, PN
    1.3-1.5:    Гос-во выдачи   RUS
    1.6-1.44:   ФИО             ФАМИЛИЯ<<ИМЯ<ОТЧЕСТВО
2.  2.1-2.9:    Серия и номер   СЕРИЯ[0:3] + НОМЕР
    2.10:       Контрольная цифра 
    2.11-2.13:  Гражданство     RUS 
    2.14-2.19:  Дата рождения   YYMMDD 
    2.20:       Контрольная цифра 
    2.21:       Пол             F, M 
    2.22-2.27:  Дата срока действия 
    2.28:       Контрольная цифра <
    2.29-2.42:  Доп инф 
    2.29:       Последняя цифра в серии паспорта 
    2.30-2.35:  Дата выдачи     YYMMDD 
    2.36-2.41:  Код подраздел 
    2.42:                       < 
    2.43:       Контрольная цифра 
    2.44:       Заключительная контрольная цифра
"""

import json
import argparse
import re
import sys
import pickle
from collections import Counter
from datetime import datetime, date
from difflib import SequenceMatcher
from pathlib import Path

import pandas as pd
import numpy as np
from scipy.stats import beta
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import KFold, StratifiedKFold

# --- Константы ---
FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date", "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number", "Issue_organization_ru"]
NOW = date(2026, 10, 1) # 1 октября 2026
TRANSLIT = {"А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E",
            "Ё": "2", "Ж": "J", "З": "Z", "И": "I", "Й": "Q", "К": "K",
            "Л": "L", "М": "M", "Н": "N", "О": "O", "П": "P", "Р": "R",
            "С": "S", "Т": "T", "У": "U", "Ф": "F", "Х": "H", "Ц": "C",
            "Ч": "3", "Ш": "4", "Щ": "W", "Ъ": "X", "Ы": "Y", "Ь": "9",
            "Э": "6", "Ю": "7", "Я": "8"}
SOURCES = ["web", "mobile", "mobile_v2"]
HERE = Path(__file__).resolve().parent
MODEL_PATH = HERE / "model.pkl"
SUBDIV_PATH = HERE / "reference" / "subdivisions.csv"
subdivisions = pd.read_csv(SUBDIV_PATH, dtype=str).groupby("code")["name"].apply(list).to_dict()
# Пороги для train
CONF_MIN = 0.9      # минимальная уверенность по любому символу VIZ/MRZ
LINE_H = (24, 38)   # допустимая высота рамки поля на одну прочитанную строку, px
GAP_MAX = 5         # допустимый зазор между рамкой поля и прочитанными строками, px
RARE_DF = 30        # основа слова 'редкая', если встречается не более чем в стольких кодах справочника

RE_DATE = re.compile(r"\d{2}\.\d{2}\.\d{4}")
RE_CODE = re.compile(r"\d{3}-\d{3}")
RE_LIC = re.compile(r"\d{2} \d{2} \d{6}")
RE_NAME = re.compile(r"[А-ЯЁ]+(?:[ -][А-ЯЁ]+)*")

# --- Даты ---

def parse_date(s: str) -> date | None:
    """ДД.ММ.ГГГГ -> date; None, если формат или календарная дата неверны."""
    if not RE_DATE.fullmatch(s or ""):
        return None
    try:
        return datetime.strptime(s, "%d.%m.%Y").date()
    except ValueError:
        return None
    
def yymmdd_to_ddmmyyyy(s: str) -> str | None:
    """Изменяет формат даты с ГГММДД в ДД.ММ.ГГГГ"""
    if not s.isdigit():
        return None

    yy, mm, dd = int(s[:2]), int(s[2:4]), int(s[4:6])
    year = 2000 + yy if 2000 + yy <= NOW.year else 1900 + yy

    try:
        return date(year, mm, dd).strftime("%d.%m.%Y")
    except ValueError:
        return None

def add_years(d: date, n: int) -> date:
    try:
        return d.replace(year=d.year + n)
    except ValueError: # 29 февраля
        return d.replace(year=d.year + n, day=28)

def passport_status(birth: date, issue: date) -> str:
    if issue > NOW:
        return "Issue in future"
    elif issue < add_years(birth, 14):
        return "Issue before 14"
    for age in (20, 45):
        if issue < add_years(birth, age) <= NOW:
            return "Expired"
    return "OK"

# --- MRZ ---

def translit(s: str) -> str:
    """Кириллица -> написание в MRZ. Все, чего нет в таблице (пробел, дефис,
    знаки, цифры, латиница), превращается в '<'."""
    return "".join(TRANSLIT.get(c, "<") for c in s.upper())

def parse_fio(mrz_top: str) -> tuple[str, str, str] | None:
    name_part = mrz_top[5:].split("<<", 1)

    if len(name_part) != 2:
        return None

    surname = name_part[0].replace("<", "-").strip("-")
    given_names = [part for part in name_part[1].strip("<").split("<") if part]

    if not surname or not given_names:
        return None
    
    name = given_names[0]
    patronymic = given_names[1] if len(given_names) > 1 else ""

    return surname, name, patronymic

def mrz_check_digit(value: str) -> int:
    weights = (7, 3, 1)
    total = 0
    for i, c in enumerate(value):
        if c == '<':
            v = 0
        elif c.isdigit():
            v = int(c)
        elif 'A' <= c <= 'Z':
            v = ord(c) - 55
        else:
            return -1

        total += v * weights[i % 3]
    return total % 10

def check_digit_check(value: str, digit: str) -> bool:
    return digit.isdigit() and mrz_check_digit(value) == int(digit)

def parse_mrz(doc: dict) -> dict | None:
    if doc['mrz'] is None:
        return None
    lines = doc['mrz']['lines']
    if len(lines) != 2:
        return None
    mrz_top = lines[0]['text']
    mrz_bot = lines[1]['text']
    if len(mrz_top) != 44 or len(mrz_bot) != 44: # сканы с MRZ 43/45 символов отсеиваем
        return None
    checks = {"ser_nom": check_digit_check(mrz_bot[:9], mrz_bot[9]),
              "birth": check_digit_check(mrz_bot[13:19], mrz_bot[19]),
              "dopoln": check_digit_check(mrz_bot[28:42], mrz_bot[42]),
              "final": check_digit_check(mrz_bot[0:10] + mrz_bot[13:20] + mrz_bot[21:43], mrz_bot[43]),}

    FIO = parse_fio(mrz_top)
    if not FIO:
        return None
    DOC_SER = f"{mrz_bot[:2]} {mrz_bot[2]+mrz_bot[28].strip('<')}"
    DOC_NUM = mrz_bot[3:9]
    LIC_NUM = f"{DOC_SER} {DOC_NUM}"
    DoB = yymmdd_to_ddmmyyyy(mrz_bot[13:19])
    SEX = mrz_bot[20]
    ISSUE_DATE = yymmdd_to_ddmmyyyy(mrz_bot[29:35])
    SUBDIV = f"{mrz_bot[35:38]}-{mrz_bot[38:41]}"
    conf_top, conf_bot = lines[0]['char_conf'], lines[1]['char_conf']
    return {
        "Last_name_ru": FIO[0],
        "First_name_ru": FIO[1],
        "Middle_name_ru": FIO[2] or None,
        "Sex_ru": {"M": "МУЖ.", "F": "ЖЕН."}.get(SEX),
        "Birth_date": DoB,
        "Birth_place_ru": None,
        "Issue_date": ISSUE_DATE,
        "Issue_organisation_code": SUBDIV,
        "Licence_number": LIC_NUM,
        "Issue_organization_ru": None,
        "checks": checks,           # проверенные цифры для решения
        "names_raw": mrz_top[5:],   # ФИО как есть, для точной сверки
        "min_conf": min(conf_top + conf_bot, default=0.0),
        # уверенность по участкам: имена / номер / дата рождения / дата выдачи и код
        "conf_names": min(conf_top[5:], default=0.0),
        "conf_num": min(conf_bot[0:10], default=0.0),
        "conf_birth": min(conf_bot[13:20], default=0.0),
        "conf_issue": min(conf_bot[28:42], default=0.0),
    }

# --- VIZ ---

def viz_text(doc: dict, field: str) -> str | None:
    value = (doc.get('viz') or {}).get(field)
    if not value or not value.get('lines'):
        return None
    text = " ".join(" ".join(line['text'].split()) 
                    for line in value['lines']).strip() or None
    return text

def extract_viz_fields(doc: dict) -> dict:
    """None - детектор поле не нашел (для отчества может быть ОК)"""
    return {f: viz_text(doc, f) for f in FIELDS}

def viz_min_conf(doc: dict, field:str) -> float:
    v = (doc.get('viz') or {}).get(field)
    if not v or not v.get('lines'):
        return 0.0
    cc = [c for l in v['lines'] for c in l['char_conf']]
    return min(cc) if cc else 0.0

def geometry(doc: dict, field: str) -> dict:
    """Геометрия поля: рамка поля против прочитанных строк.
    Потерянная строка: рамка покрывает больше места, чем занимают прочитанные строки
    (высота на строку ~50-60 px вместо ~30). char_conf этого не замечает.
    *_px - в пикселях (для жесткого правила), остальные в долях высоты строки
    (для модели: не зависят от масштаба снимка)."""
    keys = ("h_per_line", "top_px", "bot_px", "h_rel", "top_rel", "bot_rel", "max_gap_rel")
    v = (doc.get('viz') or {}).get(field)
    if not v or not v.get('lines'):
        return {k: np.nan for k in keys}
    _, y, _, h = v['box']
    boxes = sorted((l['box'] for l in v['lines']), key=lambda b: b[1])
    n = len(boxes)
    lh = (sum(b[3] for b in boxes) / n) or 1.0
    top = boxes[0][1] - y
    bot = (y + h) - max(b[1] + b[3] for b in boxes)
    gaps = [boxes[i + 1][1] - (boxes[i][1] + boxes[i][3]) for i in range(n - 1)]
    return {"h_per_line": h / n, "top_px": top, "bot_px": bot,
            "h_rel": h / (n * lh), "top_rel": top / lh, "bot_rel": bot / lh,
            "max_gap_rel": max(gaps) / lh if gaps else 0.0}

def box_ok(doc: dict, field: str) -> bool:
    g = geometry(doc, field)
    if np.isnan(g["h_per_line"]):
        return False
    return (LINE_H[0] <= g["h_per_line"] <= LINE_H[1]
            and abs(g["top_px"]) <= GAP_MAX and abs(g["bot_px"]) <= GAP_MAX)

# --- Справочник ---

def norm(s) -> str:
    return " ".join(str(s or "").split()).upper()

def stems(text: str) -> set[str]:
    """Основы значимых слов: 'КАСИМОВСКОМ' и 'КАСИМОВСКИЙ' -> 'КАСИМ'."""
    return {w[:5] for w in re.findall(r"[А-ЯЁ]{5,}", text.upper())}


STEM_DF = Counter()
for _names in subdivisions.values():
    STEM_DF.update(set().union(*(stems(n) for n in _names)))


def _key_stems(name: str) -> set[str]:
    """Ключевые основы строки справочника: самые редкие слова (район, город, регион)."""
    st = stems(name)
    for df_max in (RARE_DF, 300):
        key = {x for x in st if STEM_DF[x] <= df_max}
        if key:
            return key
    return st


def org_ok(code: str, org_text: str) -> bool:
    """Орган выдачи совпадает хотя бы с одной строкой справочника для этого кода
    по ключевым словам. Названия в паспортах и справочнике записаны по-разному
    ('ОТДЕЛ ПО ВОПРОСАМ МИГРАЦИИ ОМВД' / 'ОТДЕЛЕНИЕМ УФМС'), поэтому только по ним."""
    names = subdivisions.get(code)
    if not names:
        return False
    viz = stems(org_text)
    return any(_key_stems(n) <= viz for n in names)

# --- Словарь слов ---

def words(text: str) -> list[str]:
    """Слова текста; чисто цифровые (номера) пропускаем. Слово с примесью цифр или
    латиницы ('СТАНО3ОЙ') остается и не попадет в словарь."""
    return [w for w in re.findall(r"[А-ЯЁA-Z0-9]+", text.upper()) if not w.isdigit()]


def deletes(w: str) -> set[str]:
    return {w[:i] + w[i + 1:] for i in range(len(w))}


def make_near(vocab: set[str]) -> set[str]:
    """Индекс для поиска слов на расстоянии ~1 правки от известных (метод удалений):
    слово vocab и все его варианты с одной выпавшей буквой."""
    near = set()
    for w in vocab:
        if len(w) >= 4:
            near.add(w)
            near |= deletes(w)
    return near


REF_WORDS = {w for names in subdivisions.values() for n in names for w in words(n)}


def build_vocab(label_rows: list[dict]) -> dict:
    """label_rows: поля из train_labels. Слова из правильной разметки + слова справочника
    (города и районы из справочника годятся и для места рождения)."""
    org, bp = set(REF_WORDS), set(REF_WORDS)
    for g in label_rows:
        org.update(words(g["Issue_organization_ru"] or ""))
        bp.update(words(g["Birth_place_ru"]))
    return {"org": org, "bp": bp, "org_near": make_near(org), "bp_near": make_near(bp)}


def vocab_features(text: str, vocab: dict, key: str) -> tuple[float, float]:
    """(число неизвестных слов, из них 'подозрительных').
    Подозрительное = неизвестное, но в одной правке от известного: 'ИРКУСКОЙ' -> 'ИРКУТСКОЙ',
    'СТАНО3ОЙ' -> 'СТАНОВОЙ'. Просто новое название деревни от известных слов далеко."""
    near = vocab[key + "_near"]
    unknown = [w for w in words(text) if w not in vocab[key]]
    suspicious = [w for w in unknown if len(w) >= 4 and (w in near or deletes(w) & near)]
    return float(len(unknown)), float(len(suspicious))

# --- Сверка MRZ и VIZ ---

def format_ok(viz: dict) -> bool:
    """Формат значений как в разметке. Без этого strip(translit()) в сверке имен скрыл бы мусор."""
    if not (RE_NAME.fullmatch(viz["Last_name_ru"]) and RE_NAME.fullmatch(viz["First_name_ru"])):
        return False
    if viz["Middle_name_ru"] and not RE_NAME.fullmatch(viz["Middle_name_ru"]):
        return False
    return (viz["Sex_ru"] in ("МУЖ.", "ЖЕН.")
            and bool(RE_CODE.fullmatch(viz["Issue_organisation_code"]))
            and bool(RE_LIC.fullmatch(viz["Licence_number"])))
    
def compare_mrz_viz(mrz: dict, viz: dict) -> dict[str, bool]:
    """Словарь с проверкой совпадения значений MRZ и VIZ
    Поля, которых в MRZ нет, не сравниваются"""
    res = {}
    surname_raw, rest_raw = mrz["names_raw"].split("<<", 1)
    rest_raw = rest_raw.rstrip("<")
    last, first, mid = viz["Last_name_ru"], viz["First_name_ru"], viz["Middle_name_ru"] or ""
    exp_rest = translit(first + (" " + mid if mid else ""))
    res["Last_name_ru"] = translit(last) == surname_raw
    res["First_name_ru"] = res["Middle_name_ru"] = exp_rest == rest_raw
    for f in ("Sex_ru", "Birth_date", "Issue_date", "Issue_organisation_code", "Licence_number"):
        res[f] = viz[f] is not None and viz[f] == mrz[f]
    return res

# --- Признаки для модели ---

def build_features(doc: dict, mrz: dict, viz: dict, vocab: dict) -> dict:
    """Все проверки, которые не являются жестким отказом, превращаем в числа:
    модель взвешивает их, а порог выбирается по out-of-fold ошибкам."""
    f = {}
    q = doc.get("quality") or {}
    for k in ("sharpness", "doc_px", "glare"):
        f[k] = q.get(k, np.nan)
    for s in SOURCES:
        f[f"src_{s}"] = float(doc.get("source") == s)

    for name in FIELDS:
        v = (doc.get("viz") or {}).get(name) or {}
        confs = [c for l in (v.get("lines") or []) for c in l["char_conf"]]
        f[f"{name}_conf"] = v.get("conf", np.nan)
        f[f"{name}_min"] = min(confs, default=np.nan)
        f[f"{name}_mean"] = sum(confs) / len(confs) if confs else np.nan
        f[f"{name}_n_low"] = float(sum(c < CONF_MIN for c in confs))
        f[f"{name}_n_lines"] = float(len(v.get("lines") or []))
        for k, val in geometry(doc, name).items():
            f[f"{name}_{k}"] = val

    for k in ("min_conf", "conf_names", "conf_num", "conf_birth", "conf_issue"):
        f[f"mrz_{k}"] = mrz[k]

    code, org = viz["Issue_organisation_code"], viz["Issue_organization_ru"]
    names = subdivisions.get(code, [])
    f["ref_found"] = float(bool(names))
    f["ref_org_ok"] = float(org_ok(code, org))
    f["ref_sim"] = max((SequenceMatcher(None, norm(org), norm(n)).ratio() for n in names), default=np.nan)

    f["bp_unknown"], f["bp_suspicious"] = vocab_features(viz["Birth_place_ru"], vocab, "bp")
    f["org_unknown"], f["org_suspicious"] = vocab_features(org, vocab, "org")

    mid = viz["Middle_name_ru"]
    f["has_mid"] = float(bool(mid))
    f["name_has_sep"] = float(any(c in n for n in (viz["Last_name_ru"], viz["First_name_ru"], mid)
                                  for c in " -"))  # MRZ не различает пробел и дефис
    if mid.endswith("ИЧ"):
        f["patr_sex_ok"] = float(viz["Sex_ru"] == "МУЖ.")
    elif mid.endswith("НА"):
        f["patr_sex_ok"] = float(viz["Sex_ru"] == "ЖЕН.")
    else:
        f["patr_sex_ok"] = np.nan
    birth, issue = parse_date(viz["Birth_date"]), parse_date(viz["Issue_date"])
    f["age_at_issue"] = (issue - birth).days / 365.25
    return f

# --- Анализ одного документа ---

def analyze(doc: dict, vocab: dict) -> tuple[str | None, dict | None, dict | None]:
    """-> (причина жесткого отказа | None, признаки | None, поля | None)"""
    try:
        mrz = parse_mrz(doc)
        if mrz is None:
            return "mrz", None, None
        if not all(mrz["checks"].values()):
            return "mrz_check_digits", None, None 
        
        viz = extract_viz_fields(doc)
        if any(viz[f] is None for f in FIELDS if f != "Middle_name_ru"):
            return "viz_field_missing", None, None
        if viz["Middle_name_ru"] is None:
            viz["Middle_name_ru"] = ""
        if not format_ok(viz):
            return "bad_format", None, None
        
        compare = compare_mrz_viz(mrz, viz)
        if any(not cmp for cmp in compare.values()):
            return "mrz_not_like_viz", None, None
        
        birth, issue = parse_date(viz["Birth_date"]), parse_date(viz["Issue_date"])
        if birth is None or issue is None:
            return "bad_date", None, None
        if passport_status(birth, issue) != "OK":
            return "passport_" + passport_status(birth, issue), None, None
        
        for f in ("Birth_place_ru", "Issue_organization_ru"):
            if not box_ok(doc, f):
                return "lost_line_" + f, None, None

        return None, build_features(doc, mrz, viz, vocab), viz
    except Exception:
        return "exception", None, None
    
# --- Градиентный бустинг ---

def make_model(seed: int = 42) -> HistGradientBoostingClassifier:
    return HistGradientBoostingClassifier(max_depth=4, learning_rate=0.05, max_iter=250,
                                          min_samples_leaf=20, l2_regularization=1.0,
                                          early_stopping=False, random_state=seed)

def oof_scores(X: pd.DataFrame, y: np.ndarray, strat: np.ndarray, seed: int = 42) -> np.ndarray:
    oof = np.zeros(len(y))
    for tr, va in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strat):
        model = make_model(seed).fit(X.iloc[tr], y[tr])
        oof[va] = model.predict_proba(X.iloc[va])[:, 1]
    return oof


def threshold_for(oof: np.ndarray, y: np.ndarray, max_wrong: int = 0) -> float:
    """Наименьший порог, при котором среди принятых (оценка > порога)
    не более max_wrong документов с ошибкой."""
    wrong = np.sort(oof[y == 0])[::-1]
    return float(wrong[max_wrong]) if len(wrong) > max_wrong else 0.0


def choose_threshold(X, y, strat, repeats: int = 5, max_wrong: int = 0):
    """Порог по out-of-fold оценкам; берется наибольший по повторам CV (консервативно)."""
    oofs = [oof_scores(X, y, strat, seed) for seed in range(repeats)]
    return max(threshold_for(o, y, max_wrong) for o in oofs), oofs


def clopper_pearson_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """Верхняя граница доли ошибок (односторонняя, 1-alpha) при k ошибках из n."""
    if n == 0 or k >= n:
        return 1.0
    return float(beta.ppf(1 - alpha, k + 1, n - k))


def nested_estimate(X, y, strat, src, repeats: int, max_wrong: int, seed: int = 100):
    """Честная оценка: порог подбирается без проверяемого фолда."""
    accepted = wrong = 0
    by_src = {s: [0, 0] for s in SOURCES}
    for tr, te in StratifiedKFold(5, shuffle=True, random_state=seed).split(X, strat):
        thr, _ = choose_threshold(X.iloc[tr], y[tr], strat[tr], repeats, max_wrong)
        p = make_model(0).fit(X.iloc[tr], y[tr]).predict_proba(X.iloc[te])[:, 1]
        chosen = p > thr
        accepted += int(chosen.sum())
        wrong += int((chosen & (y[te] == 0)).sum())
        for s, bad in zip(src[te][chosen], y[te][chosen] == 0):
            by_src[s][0] += 1
            by_src[s][1] += int(bad)
    return accepted, wrong, by_src

# --- Ввод-вывод ---

def read_jsonl(path):
    return pd.read_json(path, lines=True, precise_float=True)

def write_jsonl(data: list[dict], out_path: str):
    if data[0] is None:
        return
    out_path = Path(out_path)
    pd.DataFrame(data).to_json(out_path, orient="records", lines=True, force_ascii=False)
    return 

def answer(doc_id: str, fields: dict | None) -> dict:
    if fields is None:
        return {"doc_id": doc_id, "decision": "reject"}
    return {"doc_id": doc_id, "decision": "accept", "fields": fields}

# --- Обучение ---

def run_train(pred_path, labels_path, out_path, model_path, max_wrong: int, repeats: int):
    docs = read_jsonl(pred_path).to_dict(orient="records")
    labels = read_jsonl(labels_path).set_index('doc_id')['fields'].to_dict()
    n = len(docs)
    print(f"документов: {n}")

    # Словарь строится по разметке ДРУГИХ фолдов: иначе слова правильного ответа
    # всегда оказались бы известными и признаки были бы нечестными.
    reasons, feats, fields = [None] * n, [None] * n, [None] * n
    for tr, va in KFold(5, shuffle=True, random_state=0).split(docs):
        vocab = build_vocab([labels[docs[i]["doc_id"]] for i in tr])
        for i in va:
            reasons[i], feats[i], fields[i] = analyze(docs[i], vocab)

    print("жесткий отказ по причинам:")
    for r, c in Counter(r for r in reasons if r).most_common():
        print(f"  {r:34s}{c:6d}")
    cand = [i for i in range(n) if reasons[i] is None]
    print(f"прошли жесткие правила: {len(cand)} ({len(cand) / n:.1%})")

    X = pd.DataFrame([feats[i] for i in cand])
    y = np.array([int(all(norm(fields[i][f]) == norm(labels[docs[i]["doc_id"]][f])
                          for f in FIELDS)) for i in cand])
    src = np.array([docs[i].get("source", "web") for i in cand])
    strat = y * 10 + np.array([SOURCES.index(s) if s in SOURCES else 0 for s in src])
    print(f"из них полностью верных: {y.sum()} ({y.mean():.1%}); "
          f"среди всех документов: {y.sum() / n:.1%} (потолок охвата)")

    thr, oofs = choose_threshold(X, y, strat, repeats, max_wrong)
    print("\nкомпромисс охват / ошибки (out-of-fold; k = допустимо ошибок среди принятых на train):")
    for k in (0, 1, 2, 3, 5, 10):
        t_k = max(threshold_for(o, y, k) for o in oofs)
        acc = np.mean([(o > t_k).sum() for o in oofs])
        bad = np.mean([((o > t_k) & (y == 0)).sum() for o in oofs])
        print(f"  k={k:2d}  порог {t_k:.4f}  принято {acc:6.1f} ({acc / n:5.1%})  ошибок {bad:4.1f}"
              f"  доля ошибок среди принятых {bad / max(acc, 1):.2%}")
    print(f"выбрано k={max_wrong}, порог {thr:.5f}")

    print("\nвложенная оценка (порог подбирается без проверяемого фолда)...")
    acc, bad, by_src = nested_estimate(X, y, strat, src, max(2, repeats // 2), max_wrong)
    upper = clopper_pearson_upper(bad, acc)
    print(f"  принято {acc} из {n} ({acc / n:.1%}), с ошибкой {bad}; "
          f"верхняя граница доли ошибок (Клоппер-Пирсон 95%): {upper:.3%}")
    for s, (a, w) in by_src.items():
        print(f"  {s:10s} принято {a:5d}, с ошибкой {w}")
    print(f"  прогноз для test (2000 док., состав как в train): принять ~{acc / n * 2000:.0f}, "
          f"ошибок не более ~{upper * acc / n * 2000:.1f}")

    model = make_model(0).fit(X, y)
    bundle = {"model": model, "vocab": build_vocab(list(labels.values())),
              "thr": thr, "cols": list(X.columns)}
    with open(model_path, "wb") as fh:
        pickle.dump(bundle, fh)
    print(f"\nмодель сохранена: {model_path}")

    # out-of-fold ответы по train: проверить через evaluate.py
    chosen = np.mean([o > thr for o in oofs], axis=0) >= 0.5
    accepted = {docs[i]["doc_id"]: fields[i] for j, i in enumerate(cand) if chosen[j]}
    write_jsonl([answer(d["doc_id"], accepted.get(d["doc_id"])) for d in docs], out_path)
    print(f"OOF-ответы: {out_path}")

# --- Предсказание ---

def run_predict(pred_path, out_path, model_path):
    if not Path(model_path).exists():
        sys.exit(f"нет модели {model_path}: сначала обучение (--train)")
    with open(model_path, "rb") as fh:
        bundle = pickle.load(fh)
    docs = read_jsonl(pred_path).to_dict(orient="records")
    answers, reasons = [None] * len(docs), Counter()
    cand, feats, fields = [], [], []
    for i, doc in enumerate(docs):
        reason, f, viz = analyze(doc, bundle["vocab"])
        if reason:
            reasons[reason] += 1
            answers[i] = answer(doc["doc_id"], None)
        else:
            cand.append(i)
            feats.append(f)
            fields.append(viz)
    if cand:
        X = pd.DataFrame(feats).reindex(columns=bundle["cols"])
        score = bundle["model"].predict_proba(X)[:, 1]
        for j, i in enumerate(cand):
            if score[j] > bundle["thr"]:
                answers[i] = answer(docs[i]["doc_id"], fields[j])
            else:
                reasons["low_score"] += 1
                answers[i] = answer(docs[i]["doc_id"], None)
                
    write_jsonl(answers, out_path)
    n_acc = sum(a["decision"] == "accept" for a in answers)
    print(f"принято {n_acc} из {len(docs)} ({n_acc / max(1, len(docs)):.1%})")
    for r, c in reasons.most_common():
        print(f"отказ {r:34s}{c:6d}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input", help="*_predictions.jsonl")
    ap.add_argument("output", help="answers.jsonl (при --train: OOF-ответы для evaluate.py)")
    ap.add_argument("--train", metavar="LABELS", help="train_labels.jsonl: режим обучения")
    ap.add_argument("--model", default=str(MODEL_PATH))
    ap.add_argument("--max-wrong", type=int, default=0,
                    help="сколько ошибок среди принятых допускается на train при выборе порога")
    ap.add_argument("--repeats", type=int, default=5, help="повторов CV при выборе порога")
    a = ap.parse_args()
    if a.train:
        run_train(a.input, a.train, a.output, a.model, a.max_wrong, a.repeats)
    else:
        run_predict(a.input, a.output, a.model)
        
if __name__ == "__main__":
    main()