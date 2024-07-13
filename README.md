<b>Predykcja otyłości z wykorzystaniem sieci neuronowych</b>


 <i>1. Ogólne założenia projektu</i>

Ogólnym celem projektu jest porównanie różnych modeli uczenia maszynowego oraz zbudowanie sieci neuronowej zdolnej do predykcji na podstawie podanych przez użytkownika danych, czy ma on prawidłową, czy nieprawidłową wagę, osiągając jak najwyższy możliwy procent poprawnych predykcji. Model ma pomóc w klasyfikacji pacjentów na podstawie ustalonego zestawu cech.

Aplikacja wykorzystuje do uczenia modelu ogólnodostępnego zbioru danych “Obesity Risk Dataset”. Zbiór dzielony jest na część treningową i testową, następnie wykonywane jest uczelnie modelu, oraz krosswalidacja. Użytkownik podaje dane - odpowiadając na pytania -  dla których wykonywana jest predykcja i wypisywany jest wynik.

 <i>2. Zbiór danych</i>

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
NObeyesdad (poziom otyłości) - cecha kategoryczna. Przewidywana klasa wybierana jest spośród 7 poniższych:
- Insufficient Weight - niedowaga
- Normal Weight - waga prawidłowa
- Overweight Level I - nadwaga typu I
- Overweight Level II - nadwaga typu II
- Obesity Type I - otyłość typu I
- Obesity Type II  - otyłość typu II
- Obesity Type III - otyłość typu III

 <i>3. Biblioteki</i>

Biblioteki używane w projekcie: 
- tensorflow - otwartoźródłowa biblioteka uczenia maszynowego i głębokiego uczenia. Umożliwia budowanie i trenowanie modeli sztucznej inteligencji, w tym sieci neuronowych, zarówno dla zadań klasyfikacyjnych, jak i regresyjnych.
- keras - biblioteka w języku Python, która zapewnia wysoką abstrakcję warstw, co pozwala na łatwe skonfigurowanie i użycie modeli uczenia maszynowego
- pandas - biblioteka do analizy danych w języku Python, która oferuje struktury danych i narzędzia do manipulacji tabelami (DataFrames). Pandas ułatwia wczytywanie, przetwarzanie i analizowanie danych
- numpy - biblioteka do obliczeń naukowych w Pythonie, oferująca wsparcie dla wielowymiarowych tablic oraz funkcje matematyczne i statystyczne. Jest podstawą dla wielu innych bibliotek analizy danych
- scikit-learn (sklearn) - biblioteka uczenia maszynowego w Pythonie, która oferuje różnorodne algorytmy do klasyfikacji, regresji, klasteryzacji i redukcji wymiarów, a także narzędzia do przetwarzania wstępnego danych i ewaluacji modeli.
- joblib - biblioteka w Pythonie, która jest wykorzystywana głównie do efektywnego wykonywania obliczeń, równoległego przetwarzania oraz serializacji (zapisywania i ładowania) obiektów.
- matplotlib - biblioteka do tworzenia wykresów w Pythonie. Umożliwia tworzenie szerokiej gamy wykresów, od prostych wykresów liniowych po bardziej zaawansowane wykresy trójwymiarowe.
- seaborn - biblioteka do wizualizacji danych oparta na matplotlib. Umożliwia tworzenie atrakcyjnych wizualnie i informacyjnych wykresów statystycznych

 <i>4. Pliki w projekcie</i>

- model.py - Plik zawiera model sieci neuronowej. Dane w zbiorze są najpierw przekształcane, potem dokonuje się uczenie sieci neuronowej i określane są miary jakości eksperymentu. Następnie gotowy model wraz z skalerem i encoderami zapisywane są do zewnętrznych plików.
- model_test.py - Plik zawiera porównanie różnych metod klasyfikacji i sieci neuronowej wraz z ich dokładnością. 
- prediction.py - Plik umożliwiający pobieranie danych od użytkownika, na których następnie dokonywana jest predykcja i wypisywana decyzja za pomocą nauczonego wcześniej modelu sieci neuronowej

Pliki zawierającej obiekty używane do predykcji:
 (order_of_classes.joblib, label_encoders.joblib, model.keras, scaler.joblib)

 <i>5. Struktura sieci neuronowej</i>

Sieci neuronowe w aplikacji zbudowane są za pomocą bibliotek TensorFlow oraz Keras. Podczas eksperymentów wykorzystane zostały dwa rodzaje sieci neuronowych 
- z warstwami gęstymi, oraz bardziej złożona sieć, przetwarzająca zarówno dane kategoryczne, jak i numeryczne. 

Struktura sieci z warstwami gęstymi:
- warstwa wejściowa - warstwa ma taki rozmiar, ile cech wejściowych, czyli kolumn w macierzy danych wejściowych, jest w zbiorze X 
- pierwsza warstwa ukryta - posiada funkcję aktywacji ReLU, oraz regularyzację L2 ze współczynnikiem 0.001 
- druga warstwa ukryta - posiada funkcję aktywacji ReLU, oraz regularyzację L2 ze współczynnikiem 0.001
- trzecia warstwa ukryta - używana tylko w niektórych eksperymentach, posiada tangensoidalną funkcję aktywacji oraz regularyzację L2 ze współczynnikiem 0.001
- czwarta warstwa ukryta - używana tylko w niektórych eksperymentach, posiada sigmoidalną funkcję aktywacji oraz regularyzację L2 ze współczynnikiem 0.001 
- warstwa wyjściowa - liczba neuronów jest równa liczbie klas. Posiada funkcję aktywacji softmax.
- warstwy BatchNormalization - warstwy normalizujące wyjście z poprzedniej warstwy, występują po każdej z warstw gęstych
- warstwy Dropout ze współczynnikiem 0.3 - występują po każdej z warstw gęstych, losowo wyłączają 30% neuronów

Model sieci jest modelem Keras typu Sequential, co oznacza, że warstwy są ułożone sekwencyjnie, jedna po drugiej. Po zdefiniowaniu modelu, model jest kompilowany, określany jest optymalizator, funkcja straty i metryki. Odbywa się to w każdej iteracji kroswalidacji. 
Model jest trenowany na danych treningowych przy użyciu metody fit().
Model ten używa optymalizatora Adam. Funkcja straty to funkcja categorical_crossentropy - odpowiednia dla problemów wieloklasowej klasyfikacji.

Eksperymenty zostały przeprowadzone przy ilości neuronów w warstwach ukrytych równej 64, 128 oraz 256. 

Sieć trenowana była kilkukrotnie przez 100 oraz 200 epok ze współczynnikiem uczenia 0,001, oraz rozmiarem batcha (ilości danych przekazywanych w jednej porcji do sieci) 32. 

Struktura złożonej sieci neuronowej, który przetwarza zarówno dane kategoryczne, jak i numeryczne:
- wejścia - input_categorical oraz input_numerical to dwa wejścia dla modelu. Każde z nich ma rozmiar input_size, co oznacza, że oba wejścia mają tę samą liczbę cech.
- Warstwy:
  - Warstwy dla danych numerycznych:
- dense1_numerical: Warstwa gęsta z 64 neuronami i funkcją aktywacji ReLU. Przetwarza wejście input_numerical.
- dense2_numerical: Druga warstwa gęsta z 64 neuronami i funkcją aktywacji ReLU. Przetwarza wynik z dense1_numerical.
  - Warstwa łącząca wejścia - concatenated_inputs - Łączy wejścia input_categorical i input_numerical w jedną macierz.
  - Warstwa łącząca wyniki - combined_output - Łączy wyniki z concatenated_inputs i dense2_numerical. Powstaje tu zbiór cech, który zawiera zarówno dane kategoryczne, jak i numeryczne oraz dodatkowe przetworzone dane numeryczne.
  - dense_combined -  Warstwa gęsta z 64 neuronami i funkcją aktywacji ReLU, która przetwarza połączone cechy z combined_output.
  - Warstwa wyjściowa - ostatnia warstwa gęsta z liczbą neuronów równą liczbie klas i funkcją aktywacji softmax

Po zdefiniowaniu modelu, model jest kompilowany, określany jest optymalizator, funkcja straty i metryki. Odbywa się to w każdej iteracji kroswalidacji. 
Model jest trenowany na danych treningowych przy użyciu metody fit().
Model ten używa optymalizatora Adam. Funkcja straty to funkcja categorical_crossentropy - odpowiednia dla problemów wieloklasowej klasyfikacji.

 <i>6. Metoda kroswalidacji</i>

W celu oceny wydajności modelu stosowana jest 5-krotna kroswalidacja stratyfikowana. Dane w zbiorze podzielone są na 5 równych części, każda z nich zawierająca proporcjonalną reprezentację każdej klasy. Następnie model jest trenowany na 4 z tych części i testowany na piątej. Proces ten powtarzany jest 5 razy, gdzie za każdym razem jako zbiór testowy używana jest inna część zbioru. 

 <i>7. Miary jakości</i>

Do oceny modelu stosowane są następujące miary
- dokładność (accuracy) - procent poprawnych predykcji spośród wszystkich predykcji
- F1-score - miara, która uwzględnia zarówno precyzję, jak i czułość - jest to średnia harmoniczna tych dwóch wartości
- precyzja (precision) 
- czułość (recall)

Parameter ‘weighted’ używany podczas określania tych miar oznacza, że miara jest obliczana jako średnia ważona dla wszystkich klas. Waga każdej klasy jest proporcjonalna do liczby wystąpień tej klasy w zbiorze testowym.
Średnia ważona jest obliczana z następującego wzoru:

![image](https://github.com/user-attachments/assets/b8cf2359-9bca-4a46-abdc-9dbf6dc7769b)

Gdzie:

n - liczba klas, 

wi - ilość próbek w klasie i, 

Mi - miara jakości dla klasy i, 

Średnia ważona pozwala na uwzględnienie wpływu niezrównoważonych klas na ostateczny wynik. Dzięki temu miara jest bardziej reprezentatywna dla ogólnej wydajności modelu na całym zbiorze danych, szczególnie gdy liczby próbek w różnych klasach znacząco się różnią. 

W projekcie zastosowana jest macierz pomyłek (Confusion Matrix), która pozwala zwizualizować wydajność algorytmu klasyfikacyjnego graficznie. 

Każda kolumna macierzy reprezentuje przewidywaną klasę, natomiast wiersz rzeczywisty wynik. Elementy na głównej przekątnej reprezentują liczbę poprawnej predykcji, a elementy poza główną przekątną liczbę błędnych predykcji. 

W zbiorze danych dostępnych jest 7 klas decyzyjnych, dlatego na obu osiach widoczne są liczby od 0 do 6. Odczytując macierz, z osi X wybieramy klasę przewidzianą przez sieć, natomiast z osi Y klasę rzeczywistą. Jeśli jest to ta sama klasa, to znaczy, że sieć dokonała prawidłowej predykcji. Im więcej jest wyników w danym przedziale, tym ciemniejszy jest obszar wykresu. 

 <i>8. Wynik eksperymentu</i>

Celem eksperymentu było osiągnięcie jak najwyższego możliwego procentu poprawnych predykcji oraz zastosowanie nauczonej sieci neuronowej do przewidywania decyzji na podstawie danych podanych przez użytkownika. 

Poniższe eksperymenty zostały wykonane dla sieci neuronowej stworzonej przy użyciu biblioteki Tensorflow, w szczególności jej części Keras, składającej się z kilku warstw gęstych (Dense), normalizacji wsadowej (BatchNormalization) i warstw dropout (Dropout).

Eksperymenty zostały przeprowadzone z uwzględnieniem różnej liczby neuronów warstwach ukrytych, przy różnej liczbie epok, oraz różnej architekturze sieci.

Dla każdej iteracji w czasie uczenia określana była strata w co 10 epoce, oraz finalna strata na zbiorze testowym.

Wyniki eksperymentów dostępne są w pliku Dokumentacja.pdf

 <i>9. Wnioski </i>

Model osiągnął wysoką dokładność i dobre wyniki dla każdej zastosowanej architektury. Jak pokazały wielokrotne doświadczenia, zwiększenie liczby epok oraz liczby neuronów w tych warstwach nie wpływa znacząco na dokładność sieci. Także wprowadzenie bardziej złożonego modelu sieci neuronowej nie zapewnia wyższej dokładności. Sieć neuronowa osiąga nieznacznie lepsze wyniki, jeśli trenowana jest przez więcej niż 100 epok.

Na podstawie macierzy pomyłek wnioskować można, że sieć neuronowa najlepiej rozpoznaje klasy 4 oraz 6, czyli otyłość typu 1 i typu 3.

Eksperyment wykazał, że sieć neuronowa może skutecznie klasyfikować poziomy otyłości na podstawie dostarczonych cech.





