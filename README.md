<b>Predykcja otyłości z wykorzystaniem sieci neuronowych</b>


1. Ogólne założenia projektu

Ogólnym celem projektu jest porównanie różnych modeli uczenia maszynowego oraz zbudowanie sieci neuronowej zdolnej do predykcji na podstawie podanych przez użytkownika danych, czy ma on prawidłową, czy nieprawidłową wagę, osiągając jak najwyższy możliwy procent poprawnych predykcji. Model ma pomóc w klasyfikacji pacjentów na podstawie ustalonego zestawu cech.

Aplikacja wykorzystuje do uczenia modelu ogólnodostępnego zbioru danych “Obesity Risk Dataset”. Zbiór dzielony jest na część treningową i testową, następnie wykonywane jest uczelnie modelu, oraz krosswalidacja. Użytkownik podaje dane - odpowiadając na pytania -  dla których wykonywana jest predykcja i wypisywany jest wynik.

2. Zbiór danych

Zbiór danych na których aplikacja wykonuje uczenie modelu jest zbiór “Obesity Risk Dataset”. Jest to zbiór ogólnodostępny i darmowy, który można pobrać z poniższych stron:

https://www.kaggle.com/datasets/jpkochar/obesity-risk-dataset

lub

https://archive.ics.uci.edu/dataset/544/estimation+of+obesity+levels+based+on+eating+habits+and+physical+condition

Zbiór danych zawiera dane dotyczące otyłości wśród obywatelu Mekyku, Peru i Kolumbii, określanej na podstawie ich nawyków żywieniowych oraz kondycji fizycznej. Posiada on 2111 instancji danych, oraz 16 cech. 

Opis cech na podstawie angielskiej wersji dokumentacji do zbioru danych:
- Płeć - cecha kategoryczna (Mężczyzna/Kobieta) - w zbiorze znajduje się 50% danych dotyczących kobiet i 50% dotyczących mężczyzn
- Wiek - cecha numeryczna - w zbiorze znajdują się dane dotyczące osób w wieku od 14 do 61 lat
- Wzrost - cecha numeryczna - w zbiorze znajdują się dane dotyczące osób wzrostu od 1.45 m do 1.98 m
- Waga - cecha numeryczna - w zbiorze znajdują się dane dotyczące osób posiadających wagę od 39 kg do 165 kg
- Historia otyłości w rodzinie - cecha kategoryczna, binarna (yes/no - tak/nie)
- FAVC (częste spożycie wysokokalorycznego jedzenia) - cecha kategoryczna, binarna (yes/no - tak/nie)
- FCVC (występowanie warzyw w posiłkach) - cecha numeryczna w skali 1 do 3 (1 - rzadko, 3 - często)
- NCP (ilość posiłków dziennie) - cecha numeryczna w skali od 1 do 4 (1 - 1-2 posiłki dziennie, 4 - 6-7 posiłków)
- CAEC (spożycie jedzenia pomiędzy posiłkami) - cecha kategoryczna (0 - nigdy, sometimes - czasami, frequently - często, always - zawsze)
- SMOKE (palenie papierosów) - cecha kategoryczna, binarna (yes/no - tak/nie)
- CH2O (ilość wody dziennie) - cecha numeryczna w skali 1 do 3
- SCC (monitorowanie kalorii) - cecha kategoryczna, binarna (yes/no - tak/nie)
- FAF (aktywność fizyczna - jak często) - cecha numeryczna w skali 0 do 3
- TUE (czas spędzony przed ekranem urządzeń elektronicznych - cecha numeryczna w skali 0 do 2
- CALC (spożycie alkoholu - jak często) - cecha kategoryczna (0 - nigdy, sometimes - czasami, frequently - często, always - zawsze)
- MTRANS (zazwyczaj używany środek transportu) - cecha kategoryczna, do wyboru: Public_Transportation (transport publiczny), Walking (chodzenie pieszo), Automobile (samochód), Motorbike (motocykl)

Klasa docelowa (target class):
<b>NObeyesdad (poziom otyłości)</b> - cecha kategoryczna - wynik predykcji w skali:
- Insufficient Weight - niedowaga
- Normal Weight - waga prawidłowa
- Overweight Level I - nadwaga typu I
- Overweight Level II - nadwaga typu II
- Obesity Type I - otyłość typu I
- Obesity Type II  - otyłość typu II
- Obesity Type III - otyłość typu III

4. Biblioteki

Biblioteki używane w projekcie: 
- tensorflow - otwartoźródłowa biblioteka uczenia maszynowego i głębokiego uczenia. Umożliwia budowanie i trenowanie modeli sztucznej inteligencji, w tym sieci neuronowych, zarówno dla zadań klasyfikacyjnych, jak i regresyjnych.
- keras - biblioteka w języku Python, która zapewnia wysoką abstrakcję warstw, co pozwala na łatwe skonfigurowanie i użycie modeli uczenia maszynowego
- pandas - biblioteka do analizy danych w języku Python, która oferuje struktury danych i narzędzia do manipulacji tabelami (DataFrames). Pandas ułatwia wczytywanie, przetwarzanie i analizowanie danych
- numpy - biblioteka do obliczeń naukowych w Pythonie, oferująca wsparcie dla wielowymiarowych tablic oraz funkcje matematyczne i statystyczne. Jest podstawą dla wielu innych bibliotek analizy danych
- scikit-learn (sklearn) - biblioteka uczenia maszynowego w Pythonie, która oferuje różnorodne algorytmy do klasyfikacji, regresji, klasteryzacji i redukcji wymiarów, a także narzędzia do przetwarzania wstępnego danych i ewaluacji modeli.
- joblib - biblioteka w Pythonie, która jest wykorzystywana głównie do efektywnego wykonywania obliczeń, równoległego przetwarzania oraz serializacji (zapisywania i ładowania) obiektów.

5. Pliki w projekcie

- model.py - Plik zawiera model sieci neuronowej. Dane w zbiorze są najpierw przekształcane, potem dokonuje się uczenie sieci neuronowej i określane są miary jakości eksperymentu. Następnie gotowy model wraz z skalerem i encoderami zapisywane są do zewnętrznych plików.
- model_test.py - Plik zawiera porównanie różnych metod klasyfikacji i sieci neuronowej wraz z ich dokładnością. 
- prediction.py - Plik umożliwiający pobieranie danych od użytkownika, na których następnie dokonywana jest predykcja i wypisywana decyzja za pomocą nauczonego wcześniej modelu sieci neuronowej

Pliki zawierającej obiekty używane do predykcji:
 (label_encoder_target.joblib, label_encoders.joblib, model.keras, scaler.joblib)

6. Struktura sieci neuronowej

Sieć neuronowa w aplikacji zbudowana jest za pomocą bibliotek TensorFlow oraz Keras. 
Sieć ma następującą strukturę:
- warstwa wejściowa - warstwa ma taki rozmiar, ile cech wejściowych, czyli kolumn w macierzy danych wejściowych, jest w zbiorze X 
- pierwsza warstwa ukryta - posiada 64 neurony oraz funkcję aktywacji ReLU
- druga warstwa ukryta - również posiada 64 neurony oraz funkcję aktywacji ReLU
- warstwa wyjściowa - liczba neuronów jest równa liczbie klas. Posiada funkcję aktywacji softmax.

Sieć trenowana była przez 500 epok ze współczynnikiem uczenia 0,001, oraz rozmiarem batcha (ilości danych przekazywanych w jednej porcji do sieci) 32. 

7. Metoda kroswalidacji

W celu oceny wydajności modelu stosowana jest 5-krotna kroswalidacja stratyfikowana. Dane w zbiorze podzielone są na 5 równych części, każda z nich zawierająca proporcjonalną reprezentację każdej klasy. Następnie model jest trenowany na 4 z tych części i testowany na piątej. Proces ten powtarzany jest 5 razy, gdzie za każdym razem jako zbiór testowy używana jest inna część zbioru. 

8. Miary jakości

Do oceny modelu stosowane są następujące miary
- dokładność (accuracy) - procent poprawnych predykcji spośród wszystkich predykcji
- F1-score - miara, która uwzględnia zarówno precyzję, jak i czułość - jest to średnia harmoniczna tych dwóch wartości
- precyzja (precision) 
- czułość (recall) 

Ponadto, w projekcie zastosowana jest macierz pomyłek (Confusion Matrix), która pozwala zwizualizować wydajność algorytmu klasyfikacyjnego graficznie. Każda kolumna macierzy reprezentuje przewidywaną klasę, natomiast wiersz rzeczywisty wynik. Elementy na głównej przekątnej reprezentują liczbę poprawnej predykcji, a elementy poza główną przekątną liczbę błędnych predykcji. 

9. Wynik eksperymentu

Celem eksperymentu było osiągnięcie jak najwyższego możliwego procentu poprawnych predykcji oraz zastosowanie nauczonej sieci neuronowej do przewidywania decyzji na podstawie danych podanych przez użytkownika. 

Wnioski: Model osiągnął wysoką dokładność i dobre wyniki. Być może zwiększenie liczby epok wpłynęłoby na zwiększenie dokładności, jednak byłoby to kosztem wydajności i czasu uczenia modelu, który mógłby znacznie wzrosnąć.

Eksperyment wykazał, że sieć neuronowa może skutecznie klasyfikować poziomy otyłości na podstawie dostarczonych cech.
 





