#!/usr/bin/env python3

"""
    program do wyswietlania dywanu sierpinskiego wykorzystujacy biblioteke threading, ktora ze
    wzgledu na jednowatkowosc pythona jest nieprzystosowana do obliczen rownoleglych i wolniejsza
    od podejscia sekwencyjnego, co zostanie udowodnione w tym programie.

    ```
        CPython implementation detail: In CPython, due to the Global Interpreter Lock, only one
        thread can execute Python code at once (even though certain performance-oriented libraries
        might overcome this limitation). If you want your application to make better use of the
        computational resources of multi-core machines, you are advised to use multiprocessing or
        concurrent.futures.ProcessPoolExecutor. However, threading is still an appropriate model if
        you want to run multiple I/O-bound tasks simultaneously.
    ```

    w odroznieniu od innych jezykow programowania, w jezyku python procesy sluza do rownoleglych
    obliczen, a watki do rownoleglego czekania.
"""

import functools as ft
import threading as td
import time as tm
import typing as t
import statistics as st

import numpy as np
import numpy.typing as npt
import matplotlib.pyplot as plt

Tablica: t.TypeAlias = npt.NDArray[np.float64]
Czesciowa: t.TypeAlias = t.Callable[..., t.Any]


def zeruj_kwadrat(
    koordynat_x: int,
    koordynat_y: int,
    rozmiar: int,
    macierz: Tablica,
) -> None:
    """wyzeruj pola w ksztalcie kwadratu wewnatrz mutowalnej macierzy numpy"""

    for wymiar_pierwszy in range(koordynat_x, koordynat_x + rozmiar):
        for wymiar_drugi in range(koordynat_y, koordynat_y + rozmiar):
            macierz[wymiar_pierwszy, wymiar_drugi] = 0


def dywan_sierpinskiego(
    poziom_rekurencji: int,
    mnoznik_rozmiaru: int,
    watki_wlaczone: bool,
) -> Tablica:
    """oblicz dywan sierpinskiego w sposob sekwencyjny lub w sposob zgodny z instrukcja"""

    rozmiar_calkowity: t.Final[int] = 3 ** poziom_rekurencji * mnoznik_rozmiaru
    dywan: t.Final[Tablica] = np.ones((rozmiar_calkowity, rozmiar_calkowity))

    if watki_wlaczone:
        watki: t.Final[list[td.Thread]] = []

    for poziom in range(1, poziom_rekurencji + 1):
        rozmiar_kwadratu: int = int(rozmiar_calkowity / (3 ** poziom))

        for kwadrat_x in range(0, 3 ** poziom, 3):
            koordynat_x: int = (kwadrat_x + 1) * rozmiar_kwadratu

            for kwadrat_y in range(0, 3 ** poziom, 3):
                koordynat_y: int = (kwadrat_y + 1) * rozmiar_kwadratu

                if watki_wlaczone:
                    watek: td.Thread = td.Thread(
                        target = zeruj_kwadrat,
                        args = (koordynat_x, koordynat_y, rozmiar_kwadratu, dywan)
                    )
                    watek.start()
                    watki.append(watek)
                else:
                    zeruj_kwadrat(koordynat_x, koordynat_y, rozmiar_kwadratu, dywan)

    if watki_wlaczone:
        map(td.Thread.join, watki)

    return dywan


def zmierz_czas(funkcja: t.Callable[..., t.Any]) -> float:
    """zmierz czas wykonywania funkcji"""

    start: t.Final[float] = tm.time()
    funkcja()
    koniec: t.Final[float] = tm.time()

    return koniec - start


def glowna() -> None:
    """punkt wyjscia programu"""

    rzedy: t.Final[int] = 2
    kolumny: t.Final[int] = 6
    utworz_wykres: t.Final[Czesciowa] = ft.partial(plt.subplot, rzedy, kolumny)
    ilosc_prob: t.Final[int] = 10

    def badanie_szybkosci(
        tytul: str,
        przesuniecie: int,
        poziom_rekurencji: int,
        mnoznik_rozmiaru: int,
    ) -> None:
        """wykonaj test szybkosci obliczania dywanu sierpinskiego i umiesc dane na wykresie"""

        sierpinski: t.Final[Czesciowa] = ft.partial(
            dywan_sierpinskiego,
            poziom_rekurencji,
            mnoznik_rozmiaru,
        )
        wykres: plt.Axes = utworz_wykres(przesuniecie, title = tytul)

        wykres.imshow(sierpinski(watki_wlaczone = True))

        czasy_watkowe: t.Final[list[float]] = []
        czasy_sekwencyjne: t.Final[list[float]] = []

        for proba in range(1, ilosc_prob + 1):
            czas_watkowy: float = zmierz_czas(
                lambda: sierpinski(watki_wlaczone = True),
            )
            czas_sekwencyjny: float = zmierz_czas(
                lambda: sierpinski(watki_wlaczone = False),
            )

            wykres = utworz_wykres(
                kolumny + przesuniecie,
                title = tytul,
                xlabel = "proba",
                ylabel = "czas wykonania",
            )
            wykres.bar(proba, czas_watkowy)

            czasy_watkowe.append(czas_watkowy)
            czasy_sekwencyjne.append(czas_sekwencyjny)

            print(f"{czas_watkowy = :4f}, {czas_sekwencyjny = :4f}")

        wykres.bar(ilosc_prob + 1, 0)
        wykres.bar(
            ilosc_prob + 2,
            st.mean(czasy_watkowe),
            label = "sredni czas podejsc watkowych",
            color = "red",
        )
        wykres.bar(
            ilosc_prob + 3,
            st.mean(czasy_sekwencyjne),
            label = "sredni czas podejsc sekwencyjnych",
            color = "blue",
        )
        wykres.legend()

    for przesuniecie, mnoznik_rozmiaru in enumerate((50, 100, 150), 1):
        badanie_szybkosci(
            f"mnoznik rozmiaru {mnoznik_rozmiaru}",
            przesuniecie,
            3,
            mnoznik_rozmiaru,
        )

    for przesuniecie, poziom_rekurencji in enumerate((3, 4, 5), 4):
        badanie_szybkosci(
            f"poziom rekurencji {poziom_rekurencji}",
            przesuniecie,
            poziom_rekurencji,
            10,
        )

    plt.show()


if __name__ == "__main__":
    glowna()
