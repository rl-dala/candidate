import json
import sys
import pandas as pd
import numpy as np
from collections import defaultdict
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
FIELDS = ["Last_name_ru", "First_name_ru", "Middle_name_ru", "Sex_ru", "Birth_date",   "Birth_place_ru", "Issue_date", "Issue_organisation_code", "Licence_number",   "Issue_organization_ru"]
NOW = date(2026, 10, 1) # 1 октября 2026
subdivisions = pd.read_csv(r"reference\subdivisions.csv")
TRANSLIT = {"А": "A", "Б": "B", "В": "V", "Г": "G", "Д": "D", "Е": "E",
            "Ё": "2", "Ж": "J", "З": "Z", "И": "I", "Й": "Q", "К": "K",
            "Л": "L", "М": "M", "Н": "N", "О": "O", "П": "P", "Р": "R",
            "С": "S", "Т": "T", "У": "U", "Ф": "F", "Х": "H", "Ц": "C",
            "Ч": "3", "Ш": "4", "Щ": "W", "Ъ": "X", "Ы": "Y", "Ь": "9",
            "Э": "6", "Ю": "7", "Я": "8"}
SOURCES = ["web", "mobile", "mobile_v2"]

# -----------------

def parse_date(s):
    try:
        return date.strptime(s, "%d.%m.%Y")
    except (ValueError, TypeError):
        return None
    
def add_years(d: date, n: int) -> date:
    try:
        return d.replace(year=d.year + n)
    except ValueError: # 29 февраля
        return d.replace(year=d.year + n, day=28)
    
def translit(s):
    """Кириллица -> написание в MRZ. Все, чего нет в таблице (пробел, дефис,
    знаки, цифры, латиница), превращается в '<'."""
    return "".join(TRANSLIT.get(c, "<") for c in s.upper())

def yymmdd(s: date) -> str:
    """Изменяет формат даты с ДД.ММ.ГГГГ в ГГММДД"""
    return date.strptime(s, "%d.%m.%Y").strftime("%y%m%d")

def passport_status(birth: date, issue: date) -> str:
    if issue > NOW:
        return "Issue in future"
    elif issue < add_years(birth, 14):
        return "Issue before 14"
    for age in (20, 45):
        if issue < add_years(birth, age) <= NOW:
            return "Expired"
    return "OK"

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

def viz_text(doc, field: str) -> str | None:
    if hasattr(doc, '_asdict'):
        doc = doc._asdict()
    
    value = (doc.get('viz') or {}).get(field)
    
    if not value or not value.get('lines'):
        return None
    
    text = " ".join(line['text'] 
                    for line in value.get('lines', [])
                    if line.get('text'))
    return text

def parse_mrz(doc) -> list | None:
    if hasattr(doc, '_asdict'):
        doc = doc._asdict()
    
    if doc['mrz'] is None:
        return None
    
    lines = doc['mrz']['lines']
    
    if len(lines) != 2:
        return None
    
    mrz_top = lines[0]['text']
    mrz_bot = lines[1]['text']
    
    if len(mrz_top) != 44 or len(mrz_bot) != 44:
        return None
    
    try:
        checks = {mrz_check_digit(mrz_bot[:9]) == int(mrz_bot[9]),
                  mrz_check_digit(mrz_bot[13:19]) == int(mrz_bot[19]),
                  mrz_bot[27] == "<",
                  mrz_check_digit(mrz_bot[28:42]) == int(mrz_bot[42]),
                  mrz_check_digit(mrz_bot[0:10] + mrz_bot[13:20] + mrz_bot[21:43]) == int(mrz_bot[43])}
        if not any(checks):
            return None

        FIO = [part for part in mrz_top[5:].split("<") if part.strip()]
        FIO.append("") if len(FIO) < 3 else FIO
        DOC_SER = mrz_bot[:3]+mrz_bot[28].strip('<')
        DOC_NUM = mrz_bot[3:9]
        NATION = mrz_bot[10:13]
        DoB = mrz_bot[13:19]
        SEX = mrz_bot[20]
        ISSUE_DATE = mrz_bot[29:35]
        SUBDIV = f"{mrz_bot[35:38]}-{mrz_bot[38:41]}"
        return [
            *FIO, DOC_SER, DOC_NUM, NATION, 
            DoB, SEX, ISSUE_DATE, SUBDIV,
        ]
    except (ValueError, TypeError):
        return None

def candidates(doc, field):
    c = []
    


data = pd.read_json(r"data\train_predictions.jsonl", lines=True)
for doc in data.itertuples(index=False):
    print(parse_mrz(doc))

