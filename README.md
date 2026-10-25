# Автоматизация признаков на KION

CatBoost · 30 000 пользователей · 169 946 событий · 3 seed

[Отчёт](REPORT.md) · [Дополнительные проверки](ADDITIONAL.md) · [LaTeX](report/main.tex) · [Воспроизведение](docs/REPRODUCING.md)

## Результаты

| Набор признаков | Признаки | NDCG@10 | Recall@10 |
| --- | --- | --- | --- |
| Ручной | 48 | 0.47332 | 0.71530 |
| Автоотбор | 74 | 0.48440 | 0.71821 |
| Полный банк | 128 | **0.49544** | **0.73353** |

Метрики рассчитаны для ансамблей трёх seed на одинаковом смешанном пуле кандидатов.
Автоотбор улучшает ручной набор на **0.01107**, 95% ДИ **[+0.00840, +0.01383]**.
На полном каталоге преимущество автоотбора не подтверждено, популярность за 7 дней получает более высокий NDCG

Повторения полного банка с seed 43 и 44 добавлены после исходного test и описаны в [дополнении](ADDITIONAL.md)

[Основные метрики](artifacts/main/metrics.csv) · [Повторения банка](artifacts/additional/metrics.csv) · [Интервалы](artifacts/main/bootstrap.json) · [Полный каталог](artifacts/additional/full_catalog.json)

## Запуск

Проверенная среда: Python 3.9.6, macOS arm64, CPU.
Зависимости закреплены в [requirements.txt](requirements.txt), для полного запуска рекомендуется 16-32 ГБ RAM

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make reproduce PYTHON=python
```

Команда скачивает данные, обучает модели, выполняет дополнительные проверки и формирует результаты.
Сырые данные, кэши и веса создаются при запуске и исключены из Git.
Поэтапный запуск описан в [инструкции](docs/REPRODUCING.md)

## Признаки и протокол

Ручной набор содержит метаданные, частоты, давность событий и связи пользователя с категориями контента.
Автоматический поиск добавляет временные окна, экспоненциальную память, тренды и жанровые агрегаты.
Из 11 групп выбраны `decay_2d`, `window_3d` и `window_1d`

Train, search, confirm и test разделены по времени.
Признаки используют только завершившиеся дни, выбор сохраняется до тестовой оценки.
Показов в KION нет, поэтому метрики описывают ранжирование наблюдавшихся событий среди кандидатов

[Конфигурация](configs/main.json) · [Поиск признаков](artifacts/main/search_log.json) · [Фиксация выбора](artifacts/main/selection.lock.json)

## Проверка

```sh
make check
make test PYTHON=python
make validate PYTHON=python
```

`make check` проверяет документы, контрольные суммы и пересчитывает сохранённые метрики без данных и весов.
Unit-тесты проверяют временные ограничения, кандидатов и метрики.
Полный аудит требует восстановленных данных и кэшей.
Результаты выполненных проверок сохранены в [validation.json](artifacts/main/validation.json)

## Навигация по коду

| Модуль | Назначение |
| --- | --- |
| [data.py](recsys/data.py) | Источник и полные истории выбранных пользователей |
| [candidates.py](recsys/candidates.py) | Общие кандидаты для сравниваемых моделей |
| [features.py](recsys/features.py) | Ручные признаки и банк агрегатов |
| [experiment.py](recsys/experiment.py) | Настройка CatBoost, отбор групп и оценка |
| [catalog.py](recsys/catalog.py) | Ранжирование полного доступного каталога |
| [additional.py](recsys/additional.py) | Повторения банка и контроль бюджета |
| [independent.py](recsys/independent.py) | Проверка на 5 000 новых пользователях |
| [metrics.py](recsys/metrics.py) | Метрики и bootstrap по пользователям |

Отчёт в [report/sections](report/sections/), оформление задано в [preamble.tex](report/preamble.tex).
Инструкция по сборке: [report/README.md](report/README.md)
