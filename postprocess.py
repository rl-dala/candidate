import json
import re
import sys
import pandas as pd
import numpy as np
from collections import defaultdict, Counter
from datetime import date
from typing import Any
"""
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

# --- Константы ---
FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date", "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number", "Issue_organization_ru"]
NOW = date(2026, 10, 1) # 1 октября 2026
subdivisions = pd.read_csv(r"reference\subdivisions.csv", dtype=str)
subdivisions_dict = dict(zip(subdivisions["code"], subdivisions["name"]))
TRANSLIT = {"А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E",
            "Ё": "2", "Ж": "J", "З": "Z", "И": "I", "Й": "Q", "К": "K",
            "Л": "L", "М": "M", "Н": "N", "О": "O", "П": "P", "Р": "R",
            "С": "S", "Т": "T", "У": "U", "Ф": "F", "Х": "H", "Ц": "C",
            "Ч": "3", "Ш": "4", "Щ": "W", "Ъ": "X", "Ы": "Y", "Ь": "9",
            "Э": "6", "Ю": "7", "Я": "8"}
SOURCES = ["web", "mobile", "mobile_v2"]

# Пороги для train
CONF_MIN = 0.9      # минимальная уверенность по любому символу VIZ/MRZ
LINE_H = (24, 38)   # допустимая высота рамки поля на одну прочитанную строку, px
GAP_MAX = 5         # допустимый зазор между рамкой поля и прочитанными строками, px
RARE_DF = 30        # основа слова 'редкая', если встречается не более чем в стольких кодах справочника

# --- Даты ---

def yymmdd_to_ddmmyyyy(s: str) -> str | None:
    if not s.isdigit():
        return None

    yy, mm, dd = int(s[:2]), int(s[2:4]), int(s[4:6])
    year = 2000 + yy if 2000 + yy <= NOW.year else 1900 + yy

    try:
        return date(year, mm, dd).strftime("%d.%m.%Y")
    except ValueError:
        return None

def ddmmyyyy_to_yymmdd(s: str) -> str:
    """Изменяет формат даты с ДД.ММ.ГГГГ в ГГММДД"""
    return date.strptime(s, "%d.%m.%Y").strftime("%y%m%d")

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
    if len(mrz_top) != 44 or len(mrz_bot) != 44:
        return None
    checks = {"ser_nom": check_digit_check(mrz_bot[:9], mrz_bot[9]),
                "birth": check_digit_check(mrz_bot[13:19], mrz_bot[19]),
                "dopoln": check_digit_check(mrz_bot[28:42], mrz_bot[42]),
                "final": check_digit_check(mrz_bot[0:10] + mrz_bot[13:20] + mrz_bot[21:43], mrz_bot[43]),}
    if not any(checks.values()):
        return None

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
    return {
        "Last_name_ru": FIO[0],     # латиница (как в MRZ)
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
        "min_conf": min(lines[0]['char_conf'] + lines[1]['char_conf']),
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

def labels_to_dict(doc:dict) -> dict:
    labels_dict = {f:"" for f in FIELDS}
    for field in FIELDS:
        labels_dict[field] = doc['fields'][field]
    return labels_dict

def box_ok(doc: dict, field: str) -> bool:
    """Потерянная строка: рамка поля покрывает больше места, чем занимают прочитанные
    строки (высота на строку ~50-60 px вместо ~30). char_conf этого не замечает."""
    v = (doc.get('viz') or {}).get(field)
    if not v or not v.get('lines'):
        return False
    _, y, _, h = v['box']
    lines = v['lines']
    top = min(l['box'][1] for l in lines)
    bot = max(l['box'][1] + l['box'][3] for l in lines)
    return (LINE_H[0] <= h / len(lines) <= LINE_H[1]
            and abs(top - y) <= GAP_MAX and abs((y + h) - bot) <= GAP_MAX)

# --- Справочник ---
def stems(text: str) -> set[str]:
    """Основы значимых слов: 'КАСИМОВСКОМ' и 'КАСИМОВСКИЙ' -> 'КАСИМ'."""
    return {w[:5] for w in re.findall(r"[А-ЯЁ]{5,}", text.upper())}


STEM_DF = Counter()
for _names in subdivisions_dict.values():
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
    names = subdivisions_dict.get(code)
    if not names:
        return False
    viz = stems(org_text)
    return any(_key_stems(n) <= viz for n in names)

# --- Словарь слов ---
def words(text: str) -> list[str]:
    """Слова текста; чисто цифровые (номера) пропускаем. Слово с примесью цифр или
    латиницы ('СТАНО3ОЙ') остаётся и не попадет в словарь -> отказ."""
    return [w for w in re.findall(r"[А-ЯЁA-Z0-9]+", text.upper()) if not w.isdigit()]


WORDS_ORG: set[str] = set()   # заполняются set_vocab(): слова разметки train + справочник
WORDS_BP: set[str] = set()


def set_vocab(label_rows: list[dict]):
    """label_rows: поля из train_labels. Слова из правильной разметки + слова справочника."""
    WORDS_ORG.clear()
    WORDS_BP.clear()
    ref_words = {w for names in subdivisions_dict.values() for n in names for w in words(n)}
    WORDS_ORG.update(ref_words)
    WORDS_BP.update(ref_words)      # города и районы из справочника годятся и для места рождения
    for g in label_rows:
        WORDS_ORG.update(words(g["Issue_organization_ru"]))
        WORDS_BP.update(words(g["Birth_place_ru"]))


def unknown_words(text: str, vocab: set[str]) -> list[str]:
    return [w for w in words(text) if w not in vocab]

# --- Сверка MRZ и VIZ ---
def compare_mrz_viz(mrz: dict, viz: dict) -> dict[str, bool]:
    """Словарь с проверкой совпадения значений MRZ и VIZ
    Поля, которых в MRZ нет, не сравниваются"""
    res = {}
    body = mrz["names_raw"]
    surname_raw, rest_raw = body.split("<<", 1)
    strip = lambda s: s.replace("<", "")
    last, first, mid = viz["Last_name_ru"], viz["First_name_ru"], viz["Middle_name_ru"] or ""
    res["Last_name_ru"] = strip(translit(last)) == strip(surname_raw)
    res["First_name_ru"] = res["Middle_name_ru"] = strip(translit(first + " " + mid)) == strip(rest_raw)
    for f in ("Sex_ru", "Birth_date", "Issue_date", "Issue_organisation_code", "Licence_number"):
        res[f] = viz[f] is not None and viz[f] == mrz[f]
    return res

# --- Решение ---
def decide(doc: dict) -> tuple[bool, dict | None, str]:
    mrz = parse_mrz(doc)
    if mrz is None:
        return False, None, "mrz"
    if not all(mrz["checks"].values()):
        return False, None, "mrz_check_digits"
    
    viz = extract_viz_fields(doc)
    if any(viz[f] is None for f in FIELDS if f != "Middle_name_ru"):
        return False, None, "viz_field_missing"
    if viz["Middle_name_ru"] is None:
        viz["Middle_name_ru"] = ""
    
    compare = compare_mrz_viz(mrz, viz)
    if any(not cmp for cmp in compare.values()):
        return False, None, "mrz_not_like_viz"
    
    birth, issue = date.strptime(viz["Birth_date"], "%d.%m.%Y"), date.strptime(viz["Issue_date"], "%d.%m.%Y")
    if birth is None or issue is None:
        return False, None, "bad_date"
    if passport_status(birth, issue) != "OK":
        return False, None, "passport_" + passport_status(birth, issue)
    
    if mrz["min_conf"] < CONF_MIN or any(viz_min_conf(doc, f) < CONF_MIN
                                         for f in FIELDS if viz[f]):
        return False, None, "low_conf"
    for f in ("Birth_place_ru", "Issue_organization_ru"):
        if not box_ok(doc, f):
            return False, None, "lost_line_" + f
    if not org_ok(viz["Issue_organisation_code"], viz["Issue_organization_ru"]):
        return False, None, "org_vs_reference"
    if unknown_words(viz["Issue_organization_ru"], WORDS_ORG):
        return False, None, "org_unknown_word"
    if unknown_words(viz["Birth_place_ru"], WORDS_BP):
        return False, None, "birthplace_unknown_word"
    return True, viz, ""
    
    
# --- Ввод-вывод ---
def read_jsonl(path):
    return pd.read_json(path, lines=True)

def write_jsonl(data: list[dict]):
    data = pd.DataFrame(data)
    data.to_json("answers.jsonl", orient="records", lines=True)
    return 

def main(pred_path):
    answers, reasons = [], Counter()
    data = read_jsonl(pred_path)
    for doc in data.itertuples(index=False):
        doc = doc._asdict()
        ok, fields, reason = decide(doc)
        if ok:
            answers.append({"doc_id": doc["doc_id"], "decision": "accept", "fields": fields})
        else:
            reasons[reason] += 1
    write_jsonl(answers)
    n_overall, n_acc = len(data), len(answers)
    print(f"принято {n_acc} из {n_overall} ({n_acc / n_overall:.1%})")
    for r, c in reasons.most_common():
        print(f"отказ {r:34s}{c:6d}")
        
if __name__ == "__main__":
    main(r"data\train_predictions.jsonl")

