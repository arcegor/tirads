# tirads

Файл  train.ipynb содержит команды для настройки среды обучения, а также необходимые скрипты для проверки обученной модели.
Здесь https://medium.com/swlh/guide-to-tensorflow-object-detection-tensorflow-2-e55ba3cdbc03 описывается процесс подготовки данных и обучения модели(настройки конфиг файлов, структура проекта). Процесс разметки данных, который там делается вручную, делается автоматически следующими файлами: 
Здесь ссылка на мой гугл диск с настроенным рабочим пространством https://drive.google.com/drive/folders/1rLSllIg2dCSdBr7Nqz2iIeRif7AyGGxc?usp=sharing.
Файл boxes.py выделяет координаты боксов на размеченных изображениях и сохраняет их в файл .csv. 

Пример запуска python boxes.py -i "путь до папки с исходными изображениями" -l "путь до папки с размеченными изображениями" -c "название выходного .csv файла"

Файл create_xml.py генерирует .xml файлы в паре с исходными изображениями для обучающей выборки изображений. Пример выходного .xml файла 1_1.xml есть в репозитории. 

Пример запуска python create_xml.py -x "путь до папки куда сохранить результаты" -c "название .csv файла, структура которого совпадает со структурой выходного .csv файла функции boxes.py"

Есть еще несколько .py файлов, которые были нужны для каких то манипуляций, но скорее всего их использовать не придется.



