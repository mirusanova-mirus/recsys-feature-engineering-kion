"""Render measured follow-up results, including failures, into one report section."""
import json
from pathlib import Path

import numpy as np
import pandas as pd


NAMES = {"manual": "Ручные признаки", "automatic": "Автоотбор", "all_automatic": "Весь банк", "popularity_7d": "Популярность 7 дней"}


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"] +
                     ["| " + " | ".join(map(str, r)) + " |" for r in rows])


def interval(d):
    return f"{d['delta']:+.5f} [{d['ci95'][0]:+.5f}; {d['ci95'][1]:+.5f}]"


def render(main_config=None):
    extra, replica, held = Path("artifacts/additional"), Path("artifacts/replication"), Path("artifacts/independent")
    required = [extra / "environment.json", extra / "full_catalog.json", extra / "validation.json", replica / "full_catalog.json", replica / "validation.json", held / "cohort_validation.json"]
    if not all(p.exists() for p in required):
        return None
    read = lambda folder, name: json.loads((folder / name).read_text())
    plan = read(extra, "plan.lock.json")
    expected_main = json.loads(Path(plan["fingerprint"]["config"]["main_config"]).read_text())
    if main_config is not None and main_config != expected_main:
        return None
    em, rm = pd.read_csv(extra / "metrics.csv"), pd.read_csv(replica / "metrics.csv")
    eb, rb = read(extra, "bootstrap.json"), read(replica, "bootstrap.json")
    budget = pd.read_csv(extra / "budget_metrics.csv")
    interaction = read(extra, "budget_bootstrap.json")
    ec, rc = read(extra, "full_catalog.json"), read(replica, "full_catalog.json")
    ra, rl = read(replica, "audit.json"), read(replica, "selection.lock.json")
    ee, re = read(extra, "environment.json"), read(replica, "environment.json")
    hm, hb = pd.read_csv(held / "metrics.csv"), read(held, "bootstrap.json")
    hc, ha, hl, he = [read(held,n) for n in ["full_catalog.json", "audit.json", "selection.lock.json", "environment.json"]]
    hv = read(held, "cohort_validation.json")
    held_days = [d for d in ha["days"] if d["split"] == "test"]
    held_coverage = sum(d["eligible_targets"] for d in held_days) / sum(d["raw_targets"] for d in held_days)
    bank = em[(em.regime == "mixed") & (em.variant == "all_automatic") & (em.seed == "ensemble")].iloc[0]
    repeat_rows = []
    for v in ["manual", "automatic", "all_automatic"]:
        row = em[(em.regime == "mixed") & (em.variant == v) & (em.seed == "ensemble")].iloc[0]
        seeds = em[(em.regime == "mixed") & (em.variant == v) & em.seed.isin(["42", "43", "44"])]["ndcg@10"]
        repeat_rows.append([NAMES[v], f"{row['ndcg@10']:.5f}", f"{row['recall@10']:.5f}", f"{row.roc_auc:.5f}",
                            f"{row.logloss:.5f}", f"{seeds.mean():.5f} ± {seeds.std(ddof=1):.5f}"])
    diagnostic_rows = []
    for period in ["search", "confirm"]:
        for cap in [180, 450]:
            values = [budget[(budget.period == period) & (budget.variant == v) & (budget.budget == cap) &
                              (budget.seed == "ensemble")]["ndcg@10"].iloc[0] for v in ["manual", "automatic", "all_automatic"]]
            diagnostic_rows.append([period, cap, *[f"{x:.5f}" for x in values], f"{values[2]-values[1]:+.5f}"])
    d = interaction["confirm_budget_interaction"]
    if d["ci95"][0] > 0:
        budget_text = "При увеличении бюджета отрыв банка от подмножества растёт, и интервал взаимодействия выше нуля. Это подтверждает чувствительность относительного качества к бюджету обучения в данном контроле."
    elif d["ci95"][1] < 0:
        budget_text = "При увеличении бюджета отрыв банка от подмножества уменьшается, и интервал взаимодействия ниже нуля. Поэтому объяснение провала отбора исключительно коротким обучением противоречит этому контролю."
    else:
        budget_text = "Интервал взаимодействия включает ноль: изменение относительного преимущества банка из-за одного только бюджета статистически не подтверждено. Объяснять потери отбора только 180 деревьями было бы необоснованно."
    s180 = interaction["search_bank_vs_selected_b180"]
    space_text = ("Уже на search с одинаковым коротким бюджетом весь банк сильнее выбранного набора. Конкретная слабость алгоритма: он проверяет одиночные группы и лишь top-2/top-3, но вообще не рассматривает весь банк как кандидат отбора. Ограничение пространства поиска может исключить лучшее решение даже до переноса на полный бюджет."
                  if s180["delta"] > 0 else
                  "На search с одинаковым коротким бюджетом весь банк не превосходит выбранный набор по точечной оценке. Следовательно, результат выбора зависит и от периода, на котором оцениваются сочетания; одного ранжирования групп недостаточно для гарантии переноса.")
    fold_rows, catalogue_rows = [], []
    for label, data, cat in [("16–22 августа, исходные 30k", em, ec), ("26 июля–1 августа, те же 30k", rm, rc), ("16–22 августа, другие 5k", hm, hc)]:
        for v in ["manual", "automatic", "all_automatic"]:
            row = data[(data.regime == "mixed") & (data.variant == v) & (data.seed == "ensemble")].iloc[0]
            seed_scores = data[(data.regime == "mixed") & (data.variant == v) & data.seed.isin(["42", "43", "44"])]["ndcg@10"]
            fold_rows.append([label, NAMES[v], f"{row['ndcg@10']:.5f}", f"{row['recall@10']:.5f}", f"{seed_scores.std(ddof=1):.5f}"])
        for row in cat["results"]:
            catalogue_rows.append([label, NAMES[row["variant"]], row["queries"], f"{row['ndcg@10']:.5f}", f"{row['recall@10']:.5f}"])
    rdiff = rb["mixed"]
    replication_text = ("Автоотбор улучшает ручной baseline и на втором срезе: интервал целиком выше нуля."
                        if rdiff["ci95"][0] > 0 else "На втором срезе улучшение автоотбора не подтверждается: интервал включает ноль."
                        if rdiff["ci95"][1] >= 0 else "На втором срезе автоотбор уступает ручному baseline: интервал целиком ниже нуля.")
    fold_gap = rb["mixed_bank_vs_selected"]
    fold_gap_text = ("На этом срезе банк сильнее отбора по 95% интервалу."
                     if fold_gap["ci95"][0] > 0 else "На этом срезе отбор сильнее банка по 95% интервалу."
                     if fold_gap["ci95"][1] < 0 else
                     "Разность банка и отбора на этом срезе не подтверждена: интервал включает ноль. Это не доказывает их эквивалентность, но показывает, что потерю качества от отбора нельзя объявить универсальной для всех периодов.")
    allbank_text = ("Превосходство полного банка над выбранным подмножеством сохраняется при одинаковых трёх seed."
                   if eb["mixed_bank_vs_selected"]["ci95"][0] > 0 else
                   "После трёх seed превосходство полного банка над выбранным подмножеством по 95% интервалу не подтверждено.")
    held_text = ("На новых пользователях положительная разность автоотбора подтверждается."
                 if hb["mixed"]["ci95"][0] > 0 else "На новых пользователях положительная разность автоотбора не подтверждается."
                 if hb["mixed"]["ci95"][1] >= 0 else "На новых пользователях автоотбор ухудшает качество относительно ручного baseline.")
    held_uniform = {v: hm[(hm.regime == "uniform") & (hm.variant == v) & (hm.seed == "ensemble")].iloc[0] for v in ["manual","automatic","all_automatic"]}
    uniform_text = ("На новой когорте положительная разность автоотбора подтверждается и при равномерных контролях."
                    if hb["uniform"]["ci95"][0] > 0 else "На новой когорте при равномерных контролях положительная разность автоотбора не подтверждается."
                    if hb["uniform"]["ci95"][1] >= 0 else "На новой когорте при равномерных контролях автоотбор уступает ручному baseline.")
    section = f"""## 10. Дополнительные проверки после основного эксперимента

### Статус протокола и граница выводов

Эти проверки добавлены **после просмотра исходного теста**, а не представлены как исходная регистрация. До новых запусков сохранён [plan.lock.json](artifacts/additional/plan.lock.json), время UTC `{plan['created_utc']}`. Он фиксирует варианты, seed, бюджеты, второй срез и SHA-256 исходных таблиц/результатов. Основные веса, выбранные группы, метрики и первая фиксация не менялись. Первичная разность из §1 остаётся автоотбор против ручного baseline, а не задним числом выбранный победитель.

### 10.1. Весь банк: сопоставимые повторения

Для уже определённого полного банка добавлены seed 43 и 44, без изменения параметров и признаков. Все три главных варианта теперь дополнительно сопоставлены при одинаковом ансамбле из трёх моделей. Новые предсказания seed 42 побитово совпадают с исходными сохранёнными предсказаниями.

{table(['Модель, ансамбль 3 seed', 'NDCG@10', 'Recall@10', 'ROC-AUC', 'LogLoss', 'Отдельные seed, mean ± SD'], repeat_rows)}

- Банк минус ручные, mixed: **{interval(eb['mixed_bank_vs_manual'])}**.
- Банк минус отбор, mixed: **{interval(eb['mixed_bank_vs_selected'])}**.
- Банк минус отбор, uniform: **{interval(eb['uniform_bank_vs_selected'])}**.

{allbank_text} Тем самым исходное наблюдение перестало зависеть только от одного seed. Интервалы по пользователям и SD по seed описывают разные источники вариации; три seed не заменяют независимую новую выборку.

### 10.2. Контролируемая проверка бюджета отбора

Факторный контроль: **3 фиксированных набора × 2 лимита деревьев × 3 seed**. Train, search, confirm, кандидаты, depth, регуляризация, learning rate и критерий early stopping одинаковы. Меняется только лимит 180/450; фактическое число деревьев записано в `artifacts/additional/model_records.json`. Одинаковые ранее выполненные fit загружаются точно, без обрезки моделей. Ни один из этих результатов не используется для изменения признаков или повторного выбора по тесту.

{table(['Период', 'Лимит', 'Ручные', 'Отбор', 'Банк', 'Банк − отбор'], diagnostic_rows)}

На confirm оценивается заранее зафиксированное взаимодействие:

`D = (NDCG_bank450 − NDCG_selected450) − (NDCG_bank180 − NDCG_selected180)`.

**D = {interval(d)}**, парный bootstrap по тем же пользователям и запросам, 2000 повторений. {budget_text}

{space_text} Значение bank−selected на search при 180 деревьях: **{interval(s180)}**. Это проверка механизма сбоя ограниченного поиска, а не доказательство причинной полезности каждого коррелирующего признака. Для выделения конкретных групп потребовались бы дополнительные абляции; они здесь не заявлены.

![Повторения и контроль бюджета](artifacts/additional/figures/followup.png)

### 10.3. Второй временной срез

Конфигурация: [configs/temporal_replication.json](configs/temporal_replication.json). Сохранены те же 30 000 пользователей и все их доступные исходные записи. Правило формирования кандидатов и признаков не изменено. Общие depth=6 и l2=3 перенесены из первого исследования и заморожены до новых результатов; новый перебор гиперпараметров не проводился. **Сам отбор 11 групп и top-2/top-3 выполнен заново только на более раннем search**, затем финальные модели обучены с seed 42,43,44. Вариант all-bank заранее включён как контроль.

{table(['Часть', 'Начало включено', 'Конец исключён', 'Запросы', 'Строки кандидатов'], [[s,a['start_inclusive'],a['end_exclusive'],a['queries'],a['rows']] for s,a in ra['splits'].items()])}

Новые выбранные группы: **{', '.join(rl['selected_groups']) or 'нет'}**; {len(rl['variants']['automatic'])} признаков. Решение сохранено в [replication/selection.lock.json](artifacts/replication/selection.lock.json) до чтения теста этим запуском.

{table(['Тестовая неделя', 'Модель, 3 seed', 'NDCG@10', 'Recall@10', 'SD отдельных seed'], fold_rows)}

На втором срезе автоотбор−ручные: **{interval(rdiff)}**; банк−ручные: **{interval(rb['mixed_bank_vs_manual'])}**; банк−отбор: **{interval(rb['mixed_bank_vs_selected'])}**. {replication_text} {fold_gap_text}

Тестовые недели не пересекаются. Однако это **ретроспективная временная репликация на одном источнике**, а не полностью независимый holdout: пользователи повторяются, истории вложены, а июльские исходы уже входили в train первого исследования. Кроме того, общие гиперпараметры перенесены из более позднего исследования, поэтому этот срез не является самостоятельной проспективной оценкой всей процедуры настройки. Не усредняем интервалы двух недель как независимые наблюдения. Повторение проверяет чувствительность к периоду, при этом строгая доступность истории признаков и данных обучения внутри каждого запуска сохраняется.

### 10.4. Пользователи, не участвовавшие в первом исследовании

Для более независимой проверки добавлен отдельный [user-disjoint протокол](artifacts/independent/plan.lock.json), сохранённый **до выборки новой когорты**. Из оставшихся {ha['dataset']['eligible_population_users']:,} пользователей исходной выгрузки равномерно без возвращения выбраны 5 000, seed 2027, без фильтра активности. Сохранены все **{ha['dataset']['events']:,}** предоставленных записей, а не только события интервала оценки. Пересечение с исходными 30 000: **{hv['user_overlap_with_primary']} пользователей**. `recsys.independent.audit` заново извлекает их строки из raw CSV, проверяет полное совпадение истории и воспроизводит случайную выборку.

В её тесте **{ha['splits']['test']['queries']:,} запросов / {ha['splits']['test']['users']:,} пользователей**, покрытие исходных целевых объектов **{held_coverage*100:.2f}%**. Неизвестные на дату прогноза объекты исключаются по тому же правилу, что в основной когорте; оценка условна на доступном каталоге.

Даты train/search/confirm/test совпадают с исходным августовским протоколом. Параметры заморожены на уровне первой когорты; выбор групп повторяется только на search новых пользователей. Выбраны **{', '.join(hl['selected_groups']) or 'нет'}**, {len(hl['variants']['automatic'])} признаков. Ни события, ни глобальные счётчики исходной когорты в признаки новой когорты не входят. Все три варианта заново обучены с тремя seed: оценка характеризует повторяемость процедуры с переобучением на другой когорте. Результаты приведены в общей таблице §10.3; абсолютные уровни NDCG между когортами нельзя напрямую приписывать переносу модели: наборы пользователей, каталоги и частота событий различаются, сравнение методов проводится внутри каждой когорты.

Автоотбор−ручные: **{interval(hb['mixed'])}**; банк−ручные: **{interval(hb['mixed_bank_vs_manual'])}**; банк−отбор: **{interval(hb['mixed_bank_vs_selected'])}**. {held_text} Эти пользователи не участвовали в настройке исходного исследования. Это отдельная пользовательская проверка в том же KION и календарном периоде, а не новый датасет или независимо собранная среда; общие объекты, справочники и временные факторы сохраняются.

Важный отрицательный результат на той же когорте: **uniform NDCG@10** ручных признаков **{held_uniform['manual']['ndcg@10']:.5f}**, отбора **{held_uniform['automatic']['ndcg@10']:.5f}**, банка **{held_uniform['all_automatic']['ndcg@10']:.5f}**. Автоотбор−ручные: **{interval(hb['uniform'])}**; банк−ручные: **{interval(hb['uniform_bank_vs_manual'])}**. {uniform_text} Следовательно, подтверждение на другой когорте относится к основному смешанному пулу; универсальную устойчивость к распределению контролей оно не доказывает. Интервал на {hb['mixed']['users']:,} тестовых пользователях шире, чем на основной когорте, что также ограничивает статистическую мощность.

### 10.5. Полный каталог с полным банком

Для первой недели использованы **ровно те же 210 запросов**, что и в исходной проверке; метрики ручной модели, отбора и популярности пересчитаны и совпадают. Для второй — 210 запросов с тем же заранее заданным seed 2026. На каждой неделе у моделей общий каталог и одинаковые запросы. Популярность вычисляется только из предшествующих семи дней.

{table(['Неделя', 'Модель', 'Запросы', 'NDCG@10', 'Recall@10'], catalogue_rows)}

Каталожная разность банк−популярность: первая неделя **{interval(ec['comparisons']['bank_vs_popularity'])}**, вторая **{interval(rc['comparisons']['all_automatic_vs_popularity'])}**, новые пользователи **{interval(hc['comparisons']['all_automatic_vs_popularity'])}**. Банк−ручные: первая **{interval(ec['comparisons']['bank_vs_manual'])}**, вторая **{interval(rc['comparisons']['bank_vs_manual'])}**, новые пользователи **{interval(hc['comparisons']['bank_vs_manual'])}**. Эти интервалы вторичные и исследовательские; коррекция множественных сравнений не применялась.

Каталожные результаты не использовались для выбора модели, смены закона контролей, калибровки или поправки score. Проверка добавляет сопоставимый банк и второй период, но **не выделяет причинный эффект negative sampling**. Гипотеза из §7 о распределении шума остаётся гипотезой; аналитически корректная поправка для смеси выборок без возвращения здесь не реализована.

### 10.6. Дополнительная стоимость и воспроизведение

Повторения и контроль бюджета: **{ee['new_fits']} новых fit**, {ee['logical_fits']} логических сочетаний с учётом точного переиспользования, **{ee['total_new_fit_seconds']/60:.1f} мин** нового fit. Временная репликация, включая новый поиск: **{re['model_count']} артефактов**, **{re['total_fit_seconds']/60:.1f} мин** fit; построение признаков **{ra['build_seconds']:.1f} с**. Новые пользователи: **{he['model_count']} артефактов**, **{he['total_fit_seconds']/60:.1f} мин** fit, признаки **{ha['build_seconds']:.1f} с**. Процессы запускались параллельно: 6 потоков на fit основных дополнений, 2 потока на fit когорты 5k; сумма fit не равна времени ожидания, совокупный RSS процессов не измерен. Пиковый RSS каждого процесса и все времена сохранены в их `environment.json`.

`make reproduce` включает все дополнения и проверки. Если основной запуск уже выполнен: `make additional replication independent test validate report pdf bundle`. Для новой конфигурации используйте новый output_dir. Исходная фиксация первого теста не перезаписывается дополнениями. `recsys.additional_validate` проверяет сохранность исходных SHA-256, все три seed банка, согласованность ансамблей, матрицу факторного контроля и отсутствие пересечения тестовых недель; `recsys.independent --audit-only` — непересекающуюся выборку и полноту историй.

### 10.7. Соответствие критериям и тезисы для защиты

{table(['Критерий задания', 'Проверяемое содержание решения'], [
['Сильный базовый подход', '48 ручных признаков, CatBoost с валидационной настройкой; pinned KION и все истории выбранных пользователей.'],
['Исследовательская идея', 'Типизированная грамматика 128 признаков, 11 групп и воспроизводимый wrapper-отбор без тестовых метрик.'],
['Экспериментальное сравнение', 'Единые кандидаты внутри каждого опыта, три seed, два тестовых периода, 5000 новых пользователей, полный каталог, стоимость поиска и микробенчмарк.'],
['Анализ', 'Абляции косвенных связей, факторный контроль бюджета, seed и user-bootstrap, честный отрицательный результат ограниченного отбора и пределы sampled-оценки.']])}

На защите нужно уметь объяснить: почему это не CTR; почему исключены total_dur/watched_pct; почему в историю не входит текущий день; почему будущие позитивы не удаляются из контролей; чем conditional sampled NDCG отличается от каталожного; почему важность признака не доказывает причинность; что именно автоматизировано; почему второй срез остаётся ретроспективным. Эти ограничения относятся к самой постановке, а не скрываются за высоким качеством.
"""
    # Standalone, shareable research figure; all values come from measured artifacts.
    import matplotlib.pyplot as plt
    fig_dir = extra / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    colors = ["#64748b", "#0f766e", "#2563eb"]
    for i, (v, color) in enumerate(zip(["manual", "automatic", "all_automatic"], colors)):
        values = [float(data[(data.regime == "mixed") & (data.variant == v) & (data.seed == "ensemble")]["ndcg@10"].iloc[0]) for data in [rm, em, hm]]
        axes[0].bar(np.arange(3) + (i-1)*.24, values, width=.24, label=NAMES[v], color=color)
        for j, value in enumerate(values):
            axes[0].text(j+(i-1)*.24, value+.002, f"{value:.3f}", ha="center", fontsize=8)
        vals = [float(budget[(budget.period == "confirm") & (budget.variant == v) & (budget.budget == cap) &
                             (budget.seed == "ensemble")]["ndcg@10"].iloc[0]) for cap in [180,450]]
        axes[1].plot([180,450], vals, "o-", label=NAMES[v], color=color)
    axes[0].set(xticks=[0,1,2], xticklabels=["26.07–01.08\n30k", "16.08–22.08\n30k", "16.08–22.08\nдругие 5k"], ylabel="NDCG@10", title="Тест, одинаковые ансамбли трёх seed", ylim=(0,.6))
    axes[1].set(xticks=[180,450], xlabel="Лимит деревьев", ylabel="NDCG@10 confirm", title="Контроль бюджета: фиксированные признаки")
    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(fig_dir / "followup.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    Path("ADDITIONAL.md").write_text("# Дополнительные проверки KION\n\n" + section)
    summary = f"Полный банк, ансамбль трёх seed: NDCG@10 {bank['ndcg@10']:.5f}; банк−отбор {interval(eb['mixed_bank_vs_selected'])}. {replication_text} {held_text} {uniform_text}"
    return {"section": section, "summary": summary, "bank_ndcg": float(bank["ndcg@10"])}
