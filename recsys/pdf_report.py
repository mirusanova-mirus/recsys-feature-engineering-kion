"""A standalone PDF, built from sealed empirical results (no pandas needed)."""
import argparse
import csv
from html import escape
import importlib.util
import json
from pathlib import Path
import statistics

from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether


LABELS = {"manual": "Ручные признаки", "automatic": "Автоотбор", "all_automatic": "Весь банк",
          "without_indirect": "Без косвенных связей", "static_only": "Метаданные",
          "popularity_7d": "Популярность 7d", "popularity_all": "Популярность all"}


def main(config, font_dir=None):
    out = Path(config["output_dir"])
    load = lambda name: json.loads((out / name).read_text())
    audit, lock, bootstrap, full, env, scale, records = [load(n) for n in
        ["audit.json", "selection.lock.json", "bootstrap.json", "full_catalog.json", "environment.json", "scaling.json", "model_records.json"]]
    data = audit["dataset"]
    metrics = load("metrics.json")
    def metric(v, regime="mixed", seed="ensemble"):
        return next(r for r in metrics if r["variant"] == v and r["regime"] == regime and r["seed"] == seed)
    base, auto = metric("manual"), metric("automatic")
    seed_values = {v: [metric(v, seed=str(s))["ndcg@10"] for s in config["model_seeds"]] for v in ["manual", "automatic"]}
    extra, replica, held = Path("artifacts/additional"), Path("artifacts/replication"), Path("artifacts/independent")
    followup = all((folder / "validation.json").exists() for folder in [extra, replica, held])
    if followup:
        er = lambda name: json.loads((extra / name).read_text())
        rr = lambda name: json.loads((replica / name).read_text())
        em, rm = er("metrics.json"), rr("metrics.json")
        eb, rb, bb = er("bootstrap.json"), rr("bootstrap.json"), er("budget_bootstrap.json")
        ec, rc = er("full_catalog.json"), rr("full_catalog.json")
        rl, ra = rr("selection.lock.json"), rr("audit.json")
        ee, re = er("environment.json"), rr("environment.json")
        hr = lambda name: json.loads((held / name).read_text())
        hm, hb, hc, ha, hl, he, hv = [hr(n) for n in ["metrics.json", "bootstrap.json", "full_catalog.json", "audit.json", "selection.lock.json", "environment.json", "cohort_validation.json"]]
        def follow_metric(rows, variant):
            return next(r for r in rows if r["regime"] == "mixed" and r["variant"] == variant and r["seed"] == "ensemble")
        def interval(record):
            return f"{record['delta']:+.5f} [{record['ci95'][0]:+.5f}; {record['ci95'][1]:+.5f}]"
    if font_dir is None:
        spec = importlib.util.find_spec("matplotlib")
        if spec:
            font_dir = Path(spec.origin).parent / "mpl-data/fonts/ttf"
        else:
            raise ValueError("Install matplotlib or pass --font-dir containing DejaVuSans fonts")
    font_dir = Path(font_dir)
    for name, filename in [("Research", "DejaVuSans.ttf"), ("ResearchBold", "DejaVuSans-Bold.ttf"),
                           ("ResearchMono", "DejaVuSansMono.ttf")]:
        pdfmetrics.registerFont(TTFont(name, str(font_dir / filename)))
    navy, teal, gray = colors.HexColor("#132c42"), colors.HexColor("#087f8c"), colors.HexColor("#526477")
    styles = {
        "body": ParagraphStyle("Body", fontName="Research", fontSize=9.4, leading=14, textColor=navy, spaceAfter=8),
        "small": ParagraphStyle("Small", fontName="Research", fontSize=8, leading=11, textColor=gray, spaceAfter=6),
        "title": ParagraphStyle("Title", fontName="ResearchBold", fontSize=26, leading=32, textColor=navy, spaceAfter=16),
        "h1": ParagraphStyle("H1", fontName="ResearchBold", fontSize=19, leading=24, textColor=navy, spaceAfter=14),
        "h2": ParagraphStyle("H2", fontName="ResearchBold", fontSize=12, leading=16, textColor=teal, spaceBefore=6, spaceAfter=6),
        "cell": ParagraphStyle("Cell", fontName="Research", fontSize=8, leading=11, textColor=navy),
        "head": ParagraphStyle("Head", fontName="ResearchBold", fontSize=8, leading=11, textColor=colors.white),
        "code": ParagraphStyle("Code", fontName="ResearchMono", fontSize=8, leading=12, textColor=navy, spaceAfter=8),
        "hero": ParagraphStyle("Hero", fontName="ResearchBold", fontSize=15, leading=22, textColor=teal, spaceAfter=12),
    }
    def clean(text):
        text = str(text)
        for char in "‐‑‒–—−":
            text = text.replace(char, "-")
        return text
    def p(text, style="body"):
        return Paragraph(escape(clean(text)).replace("\n", "<br/>"), styles[style])
    def tab(headers, rows, widths):
        values = [[p(x, "head") for x in headers]] + [[p(x, "cell") for x in r] for r in rows]
        result = Table(values, colWidths=widths, repeatRows=1, hAlign="LEFT")
        result.setStyle(TableStyle([("BACKGROUND", (0,0), (-1,0), navy), ("VALIGN", (0,0), (-1,-1), "TOP"),
                                   ("ROWBACKGROUNDS", (0,1), (-1,-1), [colors.HexColor("#eef5f7"), colors.white]),
                                   ("TOPPADDING", (0,0), (-1,-1), 5), ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                                   ("LINEBELOW", (0,-1), (-1,-1), .5, colors.HexColor("#c9d9df"))]))
        return result
    story = []
    add = story.append
    ci = bootstrap["mixed"]["ci95"]
    delta = bootstrap["mixed"]["delta"]
    conclusion = "95% интервал подтверждает положительную разность в основном протоколе." if ci[0] > 0 else "95% интервал не подтверждает положительную разность в основном протоколе."
    add(p("ИССЛЕДОВАНИЕ / РЕКОМЕНДАТЕЛЬНЫЕ СИСТЕМЫ", "small"))
    add(p("Автоматическое построение временных признаков", "title"))
    add(p("KION · 30 000 пользователей · воспроизводимый эксперимент", "h2"))
    add(p(f"NDCG@10: {base['ndcg@10']:.5f} → {auto['ndcg@10']:.5f}\nΔ {delta:+.5f}; 95% ДИ [{ci[0]:+.5f}, {ci[1]:+.5f}]", "hero"))
    add(p("Можно ли автоматически подобрать окна памяти и отношения между сущностями лучше ручного выбора? Сравниваются CatBoost на содержательном ручном наборе и тот же алгоритм с автоматически отобранными агрегатами."))
    allbank_summary = metric("all_automatic", seed="42")
    catalog_pop = next(r for r in full["results"] if r["variant"] == "popularity_7d")
    catalog_auto = next(r for r in full["results"] if r["variant"] == "automatic")
    add(p(f"{conclusion} Но весь банк без отбора лучше: {allbank_summary['ndcg@10']:.5f} (seed 42). На полном каталоге популярность {catalog_pop['ndcg@10']:.5f} выше автоотбора {catalog_auto['ndcg@10']:.5f}; перенос выигрыша не подтверждён. Основные значения выше - ансамбли трёх seed.", "small"))
    if followup:
        add(p(f"Дополнено: весь банк повторён с тремя seed, NDCG@10 {follow_metric(em, 'all_automatic')['ndcg@10']:.5f}. Выполнены контроль бюджета, второй временной срез и проверка на 5 000 других пользователях. Эти проверки добавлены после исходного теста и описаны в разделах 5-7.", "small"))
    add(p("Что выполнено", "h2"))
    for text in [f"Реальные данные: {data['raw_rows']:,} строк в источнике; сохранены все {data['events']:,} записей 30 000 случайных пользователей.",
                 "48 ручных признаков, банк из 128 признаков, автоматический поиск 11 групп и их сочетаний.",
                 "Четыре последовательных временных части, фиксированные кандидаты, три seed, абляции и paired bootstrap по пользователям.",
                 "Проверка полного доступного каталога для 210 случайных запросов, анализ холодных пользователей и реальные замеры стоимости."]:
        add(p("• " + text))
    add(p("Область вывода", "h2"))
    add(p("Это офлайн-ранжирование наблюдавшихся событий, а не CTR. В KION нет журнала показов, а дата last_watch_dt агрегирована по паре. Вероятности и sampled NDCG нельзя напрямую переносить на весь каталог или пользовательскую удовлетворённость."))
    add(p("24 октября 2026 · Полный текст, код и инструкции: README.md и REPORT.md", "small"))
    add(PageBreak())

    add(p("1. Данные и временной протокол", "h1"))
    add(p(f"Источник - официальный KION, data_original.zip, commit {data['source_manifest']['commit']}. Фактические даты: {data['date_min']} - {data['date_max']}. Отдельный interactions.csv.zip уменьшен до 1,59 млн строк и не использован."))
    add(p(f"Равномерная выборка пользователей без возвращения, seed 42; отбор по активности отсутствует. Медиана истории: {data['history_quantiles']['0.5']:g} записей, 90-й процентиль: {data['history_quantiles']['0.9']:g}. Истории не обрезаются. Повторных троек удалено: {data['duplicate_triples_removed']}."))
    rows = [[s, a["start_inclusive"], a["end_exclusive"], f"{a['queries']:,}"] for s,a in audit["splits"].items()]
    add(tab(["Часть", "Начало", "Конец исключён", "Запросы"], rows, [90,120,120,170]))
    add(Spacer(1,10))
    add(p("Положительная цель - все различные записи пользователя в день t. Фильм должен хотя бы раз встречаться до t в выбранных данных и ещё не встречаться у этого пользователя. Уже просмотренные объекты и неизвестные пока объекты не оцениваются."))
    add(p("Контроли: 50 уникальных объектов на запрос, 25 равномерных и 25 по прошлой популярности в степени 0.75. Положительные объекты перемешиваются с контролями. Будущие позитивы из контролей не удаляются."))
    add(p("Граница доступной информации", "h2"))
    add(p("История строго date < t; окно w = [t-w,t). Из-за дневной точности все события текущего дня недоступны. События завершённого дня обновляют историю следующего дня, включая тестовый период; веса моделей остаются фиксированными."))
    add(p("Train обучает веса. Search выбирает гиперпараметры/признаки и early stopping. Confirm проверяет выбранные решения. selection.lock.json сохраняется до чтения тестовых таблиц кодом обучения."))
    days = [d for d in audit["days"] if d["split"] == "test"]
    raw, eligible = sum(d["raw_targets"] for d in days), sum(d["eligible_targets"] for d in days)
    add(p(f"В тесте оценено {eligible:,} из {raw:,} исходных целей ({eligible/raw*100:.2f}%). Это условная оценка по известным объектам, а не end-to-end recall всех событий. Подробное покрытие каждого split и SHA-256 кандидатов сохранены в audit.json."))
    add(p("total_dur и watched_pct исключены полностью. Справочники трактуются как статические: исторических версий метаданных нет. Дата последнего просмотра не позволяет восстановить ранние повторные просмотры; строгость кода не устраняет это ограничение источника.", "small"))
    add(PageBreak())

    add(p("2. Ручной baseline и автоматизация", "h1"))
    add(p("Ручной набор", "h2"))
    add(p("Категории пользователя и объекта, item ID, год выпуска, день недели; counts пользователя/объекта за 7/30 дней и всю историю; counts и доли user-genre/country/director/type; давность, длина наблюдаемой истории и число уникальных объектов."))
    add(p("В основных категориальных агрегатах берётся первый токен. Прямой pair_count сохраняется как проверка: при фильтрации уже виденных объектов он равен нулю. User ID как признак не используется."))
    add(p(f"Выбранные параметры: {lock['params']}. Максимум {config['iterations']} деревьев, learning rate {config['learning_rate']}; LogLoss для обучения, early stopping по AUC (patience 50), выбор по NDCG@10. Настройка выполнена только на search."))
    add(p("Грамматика признаков", "h2"))
    add(tab(["Операция", "Автоматически проверяемые варианты"], [
        ["Окна counts/долей", "1, 3, 10, 14, 60 дней для шести отношений"],
        ["Экспоненциальная память", "half-life 2, 10, 30 дней; sum(2^(-age/half-life))"],
        ["Тренды", "(count_7/7) / (count_30/30 + 0.1)"],
        ["Сегментная популярность", "Возрастная группа пользователя × объект"],
        ["Все жанры", "Jaccard с последним объектом, средняя/максимальная доля жанров в 30-дневной истории"]], [140,360]))
    add(Spacer(1,10))
    add(p(f"Каждая дополнительная группа проверяется с бюджетом {config['search_iterations']} деревьев; ручной контроль получает тот же бюджет. Затем проверяются сочетания лучших двух и трёх групп. Критерий - NDCG@10, минимальный прирост {config['min_search_gain']}, не более трёх групп. Выбранный набор переобучается с полным бюджетом."))
    add(p("Выбрано: " + (", ".join(lock["selected_groups"]) or "без дополнительных групп") + f". Итого {len(lock['variants']['automatic'])} признаков. На confirm рекомендован вариант {lock['recommended_on_confirmation']}.", "hero"))
    add(p("Это ограниченный поиск в заданной человеком грамматике, связанный с идеей Deep Feature Synthesis. Он автоматизирует выбор внутри пространства, но не изобретает любые отношения или SQL-запросы. Отдельная подтверждающая валидация снижает риск переобучения поиска.", "small"))
    add(PageBreak())

    add(p("3. Результаты и неопределённость", "h1"))
    rows = []
    for v in LABELS:
        r = metric(v, seed="ensemble" if v in ["manual", "automatic"] else "42" if v not in ["popularity_7d", "popularity_all"] else "deterministic")
        rows.append([LABELS[v], f"{r['ndcg@10']:.5f}", f"{r['recall@10']:.5f}", f"{r['roc_auc']:.5f}", "-" if r['logloss'] is None else f"{r['logloss']:.5f}"])
    add(tab(["Модель", "NDCG@10", "Recall@10", "ROC-AUC", "LogLoss"], rows, [160,85,85,85,85]))
    add(Spacer(1,10))
    add(p(f"Основная разность ансамблей: {delta:+.5f}; 95% ДИ [{ci[0]:+.5f}, {ci[1]:+.5f}]. Использован парный bootstrap по пользователям, 2000 повторений, seed 2026. Все запросы пользователя переносятся вместе; метрика усредняется по запросам."))
    add(p(f"Отдельные seed 42,43,44, среднее ± SD: ручные {statistics.mean(seed_values['manual']):.5f} ± {statistics.stdev(seed_values['manual']):.5f}; авто {statistics.mean(seed_values['automatic']):.5f} ± {statistics.stdev(seed_values['automatic']):.5f}. Ансамбль - средняя вероятность. Прочие строки - seed 42 или популярность; абляции сравниваются при одинаковом seed. ДИ условен на данном обучении и наборе данных.", "small"))
    add(p("Устойчивость к кандидатам", "h2"))
    rows = []
    for regime, label in [("mixed", "Смешанные"), ("uniform", "Равномерные")]:
        for v in ["manual", "automatic"]:
            r = metric(v, regime)
            rows.append([label, LABELS[v], f"{r['ndcg@10']:.5f}", f"{r['recall@10']:.5f}"])
    for r in full["results"]:
        rows.append(["Весь каталог", LABELS[r['variant']], f"{r['ndcg@10']:.5f}", f"{r['recall@10']:.5f}"])
    add(tab(["Режим", "Модель", "NDCG@10", "Recall@10"], rows, [120,190,95,95]))
    add(Spacer(1,10))
    add(p(f"Каталог: {len(full['chosen_qids'])} случайных запросов, без сэмплирования контролей. Авто - ручные: Δ={full['bootstrap']['delta']:+.5f}; ДИ [{full['bootstrap']['ci95'][0]:+.5f}, {full['bootstrap']['ci95'][1]:+.5f}]. Положительное различие не подтверждено; популярность описательно сильнее обеих моделей. Это каталог выбранной выборки."))
    add(p("NDCG использует бинарную релевантность и все цели дня; ties разрешаются по item ID. Sampled NDCG не оценивает без смещения каталожный NDCG. ROC-AUC и LogLoss относятся к искусственному распределению кандидатов; калиброванный CTR из них не следует.", "small"))
    add(PageBreak())

    add(p("4. Что сработало и почему", "h1"))
    search = load("search_log.json")
    reference = next(r["ndcg@10"] for r in search if r["name"] == "search_reference")
    single = sorted([r for r in search if r["stage"] == "feature_screen" and r["name"] != "search_reference"], key=lambda r: -r["ndcg@10"])
    add(tab(["Одиночная группа", "NDCG@10 search", "Δ к тому же бюджету"],
            [[r["name"].replace("screen_", ""), f"{r['ndcg@10']:.5f}", f"{r['ndcg@10']-reference:+.5f}"] for r in single[:3] + single[-1:]], [230,130,140]))
    add(Spacer(1,10))
    add(p("Поиск групп на 180 деревьях не выбрал лучший набор для полного бюджета: весь банк оказался сильнее. Коррелирующие окна и взаимодействия могут объяснять это, но причинная польза отдельных групп не установлена. Таблица описывает search, а не независимые тестовые эффекты."))
    m42, a42 = metric("manual", seed="42"), metric("automatic", seed="42")
    indirect, allbank = metric("without_indirect", seed="42"), metric("all_automatic", seed="42")
    add(p(f"При seed 42 вклад косвенных исторических связей относительно абляции: {m42['ndcg@10']-indirect['ndcg@10']:+.5f} NDCG@10. Отобранный набор относительно всего банка: {a42['ndcg@10']-allbank['ndcg@10']:+.5f}. Для этих описательных разностей отдельные ДИ не вычислялись."))
    add(p("Холодные пользователи", "h2"))
    add(tab(["Модель", "Warm NDCG@10", "Cold NDCG@10", "Cold запросов"],
            [[LABELS[v], f"{r['warm_ndcg@10']:.5f}", f"{r['cold_ndcg@10']:.5f}", r["cold_queries"]] for v,r in [("manual",base),("automatic",auto)]], [170,110,110,110]))
    add(Spacer(1,10))
    add(p("Cold означает отсутствие событий до даты запроса. Для такого пользователя история не помогает; остаются справочники, календарь и глобальная популярность. Пользователи с короткой историей не удалялись, поэтому оценка не ограничена только активными клиентами."))
    add(p("Самые важные признаки автоматической модели", "h2"))
    with (out / "importance_automatic_42.csv").open() as f:
        importance = list(csv.DictReader(f))[:4]
    add(tab(["Признак", "Важность"], [[r["feature"], f"{float(r['importance']):.2f}"] for r in importance], [390,110]))
    add(Spacer(1,8))
    add(p("PredictionValuesChange описывает использование признаков, а не причинность. Для согласованного анализа нужны абляции, вариация seed и независимый период. Даже выигрыш на тесте не доказывает рост пользовательской удовлетворённости.", "small"))
    add(p("Провал переноса на каталог может быть связан с законом контролей: классификатор различает позитивы и популярностный шум, а не оценивает каталожную вероятность события. Коррекция сэмплирования не применялась; причинный эффект распределения контролей не выделен.", "small"))
    add(PageBreak())

    if followup:
        add(p("5. Повторения и контроль бюджета", "h1"))
        add(p("Дополнительный протокол сохранён до новых запусков в artifacts/additional/plan.lock.json. Исходный тест уже был известен, поэтому это post-hoc проверки; первоначальное сравнение и его результаты сохранены. Новый победитель не объявляется задним числом.", "small"))
        add(p("Полный банк: одинаковые три seed", "h2"))
        rows = []
        for v in ["manual", "automatic", "all_automatic"]:
            r = follow_metric(em, v)
            values = [x["ndcg@10"] for x in em if x["regime"] == "mixed" and x["variant"] == v and x["seed"] in ["42","43","44"]]
            rows.append([LABELS[v], f"{r['ndcg@10']:.5f}", f"{r['recall@10']:.5f}", f"{statistics.mean(values):.5f} ± {statistics.stdev(values):.5f}"])
        add(tab(["Модель, ансамбль", "NDCG@10", "Recall@10", "Отдельные seed, mean ± SD"], rows, [145,95,95,165]))
        add(Spacer(1,8))
        add(p("Банк - ручные: " + interval(eb["mixed_bank_vs_manual"]) + ". Банк - отбор: " + interval(eb["mixed_bank_vs_selected"]) + ". ДИ: paired user-bootstrap, 2000 повторений. Банк seed 42 побитово воспроизводит исходные предсказания.", "small"))
        add(p("3 набора × 2 лимита × 3 seed", "h2"))
        with (extra / "budget_metrics.csv").open() as f:
            bm = list(csv.DictReader(f))
        rows = []
        for period in ["search", "confirm"]:
            for cap in ["180", "450"]:
                vals = [float(next(x for x in bm if x["period"] == period and x["budget"] == cap and x["variant"] == v and x["seed"] == "ensemble")["ndcg@10"]) for v in ["manual", "automatic", "all_automatic"]]
                rows.append([period, cap, *[f"{v:.5f}" for v in vals], f"{vals[2]-vals[1]:+.5f}"])
        add(tab(["Период", "Лимит", "Ручные", "Отбор", "Банк", "Банк - отбор"], rows, [85,55,85,85,85,105]))
        add(Spacer(1,8))
        add(p("Данные, признаки, кандидаты, параметры и early stopping зафиксированы; меняется только лимит 180/450. На confirm проверяем D = (bank450 - selected450) - (bank180 - selected180). D = " + interval(bb["confirm_budget_interaction"]) + ".", "small"))
        interaction = bb["confirm_budget_interaction"]
        if interaction["ci95"][0] > 0:
            reason = "Больший лимит усиливает отрыв банка; относительное качество чувствительно к бюджету. Это поддерживает гипотезу о несовпадении короткого поиска и полного обучения."
        elif interaction["ci95"][1] < 0:
            reason = "Больший лимит уменьшает отрыв банка. Объяснение потерь отбора только коротким обучением противоречит этому контролю."
        else:
            reason = "Интервал взаимодействия включает ноль. Объяснять потери отбора одним только коротким обучением нельзя: этот механизм не подтверждён."
        add(p(reason))
        add(p("Банк - отбор на search при 180: " + interval(bb["search_bank_vs_selected_b180"]) + ". Поиск рассматривает лишь одиночные группы и top-2/top-3, исключая весь банк из кандидатов. Ограниченное пространство может пропустить полезное сочетание. Диагностический контроль не используется для нового выбора по тесту.", "small"))
        add(PageBreak())

        add(p("6. Повторение на другом периоде", "h1"))
        add(p("Тот же источник, 30 000 пользователей и полный набор предоставленных записей. Train 14.06-11.07, search 12-18.07, confirm 19-25.07, test 26.07-01.08. Все даты 2021 года. Веса учатся заново; внутри каждого дня доступна только прошлая история.", "small"))
        add(p("Depth=6 и l2=3 заморожены заранее. Отбор 11 групп повторён на раннем search. Выбраны " + ", ".join(rl["selected_groups"]) + f"; {len(rl['variants']['automatic'])} признаков. Решение sealed до чтения нового test. Все три варианта получают три seed.", "small"))
        add(tab(["Модель", "NDCG@10, новый test", "Recall@10"], [[LABELS[v], f"{follow_metric(rm,v)['ndcg@10']:.5f}", f"{follow_metric(rm,v)['recall@10']:.5f}"] for v in ["manual","automatic","all_automatic"]], [190,170,140]))
        add(Spacer(1,8))
        add(p("Авто - ручные: " + interval(rb["mixed"]) + ". Банк - ручные: " + interval(rb["mixed_bank_vs_manual"]) + ". Банк - отбор: " + interval(rb["mixed_bank_vs_selected"]) + ".", "small"))
        add(p("Полный каталог: обе недели", "h2"))
        rows = []
        for label, record in [("16-22.08",ec),("26.07-01.08",rc)]:
            for r in record["results"]:
                rows.append([label, LABELS[r["variant"]], f"{r['ndcg@10']:.5f}", f"{r['recall@10']:.5f}"])
        add(tab(["Неделя", "Модель, 210 запросов", "NDCG@10", "Recall@10"], rows, [90,220,95,95]))
        add(Spacer(1,8))
        add(p("Банк - популярность: 16-22.08 " + interval(ec["comparisons"]["bank_vs_popularity"]) + "; 26.07-01.08 " + interval(rc["comparisons"]["all_automatic_vs_popularity"]) + ". Каталожные score не корректировались и не использовались для настройки.", "small"))
        add(p("Тестовые недели не пересекаются, но пользователи и истории повторяются; июльские исходы входили в первый train, а гиперпараметры перенесены из более позднего исследования. Это ретроспективная проверка чувствительности к периоду, не независимый проспективный holdout. Интервалы недель не объединяются как независимые. Вторичные проверки без поправки на множественные сравнения.", "small"))
        add(PageBreak())

        add(p("7. Другие пользователи: отдельный контроль", "h1"))
        add(p("План зафиксирован до выборки в artifacts/independent/plan.lock.json. Из 932 179 пользователей вне исходной когорты равномерно без возвращения выбраны 5 000, seed 2027. Нет фильтра активности. Пересечение с исходными 30 000: ноль пользователей."))
        add(p(f"Сохранены все {ha['dataset']['events']:,} предоставленных записей. Аудит заново извлекает их из raw CSV и проверяет случайную выборку. Даты исходного августовского протокола сохранены; все split содержат только новую когорту. В тесте {ha['splits']['test']['queries']:,} запросов и {ha['splits']['test']['users']:,} пользователей, объекты вне известного каталога исключены.", "small"))
        add(p("Общие параметры заморожены до новых результатов. Отбор выполнен заново на search новой когорты: " + ", ".join(hl["selected_groups"]) + f"; {len(hl['variants']['automatic'])} признаков. Все варианты - три seed. Счётчики и каталог строятся только по новой когорте; события исходных пользователей не используются.", "small"))
        add(tab(["Модель, 3 seed", "NDCG@10", "Recall@10", "ROC-AUC"], [[LABELS[v], f"{follow_metric(hm,v)['ndcg@10']:.5f}", f"{follow_metric(hm,v)['recall@10']:.5f}", f"{follow_metric(hm,v)['roc_auc']:.5f}"] for v in ["manual","automatic","all_automatic"]], [185,105,105,105]))
        add(Spacer(1,8))
        add(p("Авто - ручные: " + interval(hb["mixed"]) + ". Банк - ручные: " + interval(hb["mixed_bank_vs_manual"]) + ". Банк - отбор: " + interval(hb["mixed_bank_vs_selected"]) + ".", "small"))
        add(p("Ограничение: на uniform-кандидатах этой когорты авто - ручные " + interval(hb["uniform"]) + "; банк - ручные " + interval(hb["uniform_bank_vs_manual"]) + ". Подтверждение в смешанном пуле не доказывает устойчивость к любому распределению контролей; исходы uniform также включены в метрики.", "small"))
        add(p("Полный каталог новых пользователей", "h2"))
        add(tab(["Модель, 210 запросов", "NDCG@10", "Recall@10"], [[LABELS[r['variant']],f"{r['ndcg@10']:.5f}",f"{r['recall@10']:.5f}"] for r in hc["results"]], [270,115,115]))
        add(Spacer(1,8))
        add(p("Банк - популярность: " + interval(hc["comparisons"]["all_automatic_vs_popularity"]) + ". Интервалы вторичные, без поправки на множественные сравнения; каталог не используется для нового выбора модели.", "small"))
        add(p("Новые пользователи не участвовали в настройке исходного исследования. Это более отдельная пользовательская проверка, однако источник, объекты и календарь остаются общими. Каталоги двух когорт различаются; абсолютные NDCG между когортами нельзя трактовать как чистый эффект переноса. Сравнения методов проводятся внутри когорты.", "small"))
        add(PageBreak())

    add(p(("8" if followup else "5") + ". Стоимость, воспроизведение и пределы", "h1"))
    add(p(f"Основные таблицы признаков: {audit['build_seconds']:.1f} с. Сумма fit вариантов: {env['total_fit_seconds']/60:.1f} мин. Пиковый RSS основного процесса: {env['peak_rss_mib']:.0f} МиБ. CPU, {env['threads']} потоков на обучение. Две абляции считались параллельно; общий RSS процессов не измерялся. Версии - environment.json. Стоимость поиска включена."))
    if followup:
        add(p(f"Дополнения: {ee['new_fits']} новых fit, {ee['total_new_fit_seconds']/60:.1f} мин. Временная репликация: {re['model_count']} артефактов, {re['total_fit_seconds']/60:.1f} мин fit. Новые 5k пользователей: {he['model_count']} артефактов, {he['total_fit_seconds']/60:.1f} мин fit. Процессы запускались параллельно: 6 потоков на fit основных дополнений, 2 у новой когорты. Сумма fit не равна времени ожидания; совокупный RSS не измерен.", "small"))
    add(p("Однодневный микробенчмарк (31.07.2021, один повтор)", "h2"))
    add(tab(["Пользователи", "История целиком", "Строки кандидатов", "Признаки, с", "Frame, МиБ"],
            [[r["sample_users"],r["complete_history_rows"],r["candidate_rows"],f"{r['feature_seconds']:.2f}",f"{r['frame_mib']:.1f}"] for r in scale["results"]], [90,110,110,95,95]))
    add(Spacer(1,10))
    add(p("Pandas повторно группирует историю по дням и окнам; реализация не готова к миллиардам событий. Нужны инкрементальные оконные счётчики и состояния затухания, SQL/streaming-агрегации, партиционирование и retrieval перед ранжированием. Время на миллиардах событий не экстраполировалось."))
    add(p("Воспроизведение", "h2"))
    add(p("python3 -m venv .venv\nsource .venv/bin/activate\npython -m pip install -r requirements.txt\nmake reproduce PYTHON=python", "code"))
    add(p("Нужны интернет для pinned KION и зависимости. Raw data, кэши, модели и предсказания остаются локально; компактный архив содержит код, отчёты и проверяемые результаты. При изменении конфигурации используйте новый output_dir. unit tests и recsys.validate проверяют временные границы, будущие события, кандидатов, метрики и checksums."))
    add(p("Ограничения", "h2"))
    add(p(("Один датасет и два ретроспективных временных среза; " if followup else "Один датасет и временной срез; ") + "30 000 пользователей; неизвестные объекты исключены; last_watch_dt агрегирован; справочники не имеют as-of снимков; нет показов и достоверных негативов; источнику добавлен шум. Полный каталог проверен на подмножестве запросов. Yambda и альтернативные нейросети не исследовались. Нужны независимый период и затем онлайн-проверка.", "small"))
    add(p("Источники", "h2"))
    for text,url in [("Petrov et al., MTS Kion Dataset, CARS@RecSys 2022", "https://github.com/irsafilo/KION_DATASET"),
                     ("Kanter & Veeramachaneni, Deep Feature Synthesis, 2015", "https://groups.csail.mit.edu/EVO-DesignOpt/groupWebSite/uploads/Site/DSAA_DSM_2015.pdf"),
                     ("Prokhorenkova et al., CatBoost, NeurIPS 2018", "https://proceedings.neurips.cc/paper/2018/hash/14491b756b3a51daac41c24863285549-Abstract.html"),
                     ("Krichene & Rendle, On Sampled Metrics, KDD 2020", "https://research.google/pubs/on-sampled-metrics-for-item-recommendation/")]:
        add(Paragraph(f'<link href="{url}" color="#087f8c">{escape(text)}</link>', styles["small"]))
    destination = Path("output/pdf/kion_feature_research.pdf")
    destination.parent.mkdir(parents=True, exist_ok=True)
    def decorate(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#d5e2e7"))
        canvas.line(47, A4[1]-35, A4[0]-47, A4[1]-35)
        canvas.setFont("Research", 7.5)
        canvas.setFillColor(gray)
        canvas.drawString(47, 26, "KION / Автоматизация признаков / 24.10.2026")
        canvas.drawRightString(A4[0]-47, 26, str(doc.page))
        canvas.restoreState()
    doc = SimpleDocTemplate(str(destination), pagesize=A4, rightMargin=47, leftMargin=47,
                            topMargin=48, bottomMargin=45, title="Автоматизация временных признаков: KION", author="Воспроизводимое исследование")
    doc.build(story, onFirstPage=decorate, onLaterPages=decorate)
    from pypdf import PdfReader
    reader = PdfReader(str(destination))
    text_lengths = [len(page.extract_text()) for page in reader.pages]
    record = {"pages": len(reader.pages), "text_characters_by_page": text_lengths,
              "unicode_replacement_characters": sum(page.extract_text().count("\ufffd") for page in reader.pages)}
    (out / "pdf_validation.json").write_text(json.dumps(record, indent=2))
    print(f"Created {destination}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/main.json")
    parser.add_argument("--font-dir")
    args = parser.parse_args()
    main(json.loads(Path(args.config).read_text()), args.font_dir)
