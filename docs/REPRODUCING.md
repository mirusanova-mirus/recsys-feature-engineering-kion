# Воспроизведение KION

[README](../README.md) · [Конфигурация](../configs/main.json) · [Среда запуска](../artifacts/main/environment.json)

## Полный запуск

Команды выполняются из корня репозитория

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
make reproduce PYTHON=python
```

Для первого скачивания нужен доступ к GitHub.
Источник и контрольные суммы закреплены в [SOURCE_MANIFEST.json](../data/SOURCE_MANIFEST.json).
Суммарное время fit основного опыта составило 62.6 мин на CPU, подготовка и оценка сюда не входят

Перед новым обучением поставляемые результаты переносятся в `artifacts/reference_snapshot_N`.
Этот шаг выполняется один раз по маркеру `artifacts/REFERENCE_RESULTS_ONLY.json`.
Кэши последующих запусков используются повторно при совпадении данных и конфигурации

## Поэтапный запуск

Перед первым запуском по этапам сохраните поставляемые результаты

```sh
make prepare-reference PYTHON=python
make download features train catalog benchmark PYTHON=python
make additional replication independent PYTHON=python
make test validate PYTHON=python
```

| Команда | Содержание |
| --- | --- |
| `make train` | Подбор CatBoost, поиск групп, три seed и абляции |
| `make catalog` | Полный доступный каталог на 210 запросах |
| `make benchmark` | Замеры на 1 000, 10 000 и 30 000 пользователей |
| `make additional` | Фиксация плана, повторения банка и бюджет 180/450 деревьев |
| `make replication` | Повторный поиск на более раннем временном срезе |
| `make independent` | Поиск и обучение на 5 000 новых пользователях |

`make reproduce` обновляет дополнительный план для нового основного запуска и сохраняет прежние дополнительные результаты в `artifacts/archive/`.
При поэтапном повторении после нового основного обучения выполните `python -m recsys.additional --seal-only --renew-plan` перед `make additional`

## Конфигурации и результаты

Основные результаты находятся в `artifacts/main/`.
Дополнительные опыты записываются в `artifacts/additional/`, `artifacts/replication/` и `artifacts/independent/`

При изменении признаков или параметров задайте новый `output_dir` в копии конфигурации.
Основные этапы принимают её через `CONFIG=configs/new.json`.
Дополнительные проверки используют отдельные планы, полный `make reproduce` рассчитан на поставляемый основной дизайн

## Отчёты

Редактируемый LaTeX собирается отдельно

```sh
make latex
```

`make report` формирует машинную версию Markdown в `output/generated/`.
Отредактированные README и отчёты в корне сохраняются, относительные ссылки в машинной версии пересчитываются

`make check` проверяет компактные результаты и документы без скачивания данных и обучения.
`make bundle` создаёт `submission/recsys_kion.zip` с кодом, результатами, Markdown и исходниками LaTeX.
Архив не содержит данные, веса, большие кэши и прежние PDF-экспорты

После нового обучения перенесите изменившиеся числа и рисунки из машинного отчёта в редактируемые Markdown и LaTeX перед упаковкой

Проверка контрольных сумм собранного архива:

```sh
python3 tools/package_submission.py --verify submission/recsys_kion.zip
```
