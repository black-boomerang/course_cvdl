# Отчёт

## Лучшая модель

Обучение проводилось при помощи библиотеки [mmdet от OpenMMLab](https://mmdetection.readthedocs.io/en/latest/).
Лучшее качество показала anchor-free модель [YOLOX-X](https://arxiv.org/pdf/2107.08430.pdf) с бэкбоуном
[CSPDarknet](https://arxiv.org/pdf/2004.10934v1.pdf), [Spatial Pyramid Pooling layer](https://arxiv.org/pdf/1406.4729.pdf),
[Path Aggregation Network](https://arxiv.org/pdf/1803.01534.pdf), отдельными головами для классификации и локализации.
В качестве аугментаций использовались Mosaic и MixUp, предложенные авторами статьи,  а также случайные повороты, сдвиги и
масштабирование. В качестве лосса используется взвешенная сумма кросс-энтропии (для классификации и локализации центра рамки) и
IoU Loss (для размера рамки). Для более качественного обнаружения объектов вместо сопоставления 
GT и предсказаний по порогу IoU, используется [Optimal Transport Assignment](https://arxiv.org/pdf/2103.14259.pdf),
который для ускорения модели был адаптирован в SimOTA.

Логи обучения лучшей модели находятся в ноутбуке train_yolox_extra.ipynb.

## Воспроизведение результатов

Для повторения лучшего эксперимента необходимо запустить train_yolox.ipynb, предварительно указав в переменной
`data_path` путь к папке с данными (в которой находится cocotext.v2.json и train2014).

## Эксперименты

Модель YOLOX была выбрана на основе анализа результатов моделей, представленных в mmdet, на других датасетах по
обнаружению объектов на изображениях. В ходе экспериментов были обучены три конфигурации YOLOX: Small, Large и Extra
Large. Ожидаемо, последняя оказалась лучше. Логи обучения двух других конфигураций представлены в ноутбуках
train_yolox_small.ipynb и train_yolox_large.ipynb. Все конфигурации отличались только размером бэкбоуна.

