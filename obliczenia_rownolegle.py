#!/usr/bin/env python3

"""
    program badajacy szybkosc wykonywania obliczen rownoleglych
    
    wielordzeniowy kod zamieszczony w instrukcji do zadania wykonuje sie wolniej niz
    jednordzeniowy, co jest zapewne zaskoczeniem dla osob czytajacych. aby temu zaradzic w tym
    programie zostala uzyta krotsza lista i bardziej wymagajaca obliczeniowo funkcja.
"""

from __future__ import annotations

import enum as en
import functools as ft
import time as tm
import typing as t

import joblib as jl
import matplotlib.pyplot as plt


class Podejscie(en.StrEnum):
    """wyliczenie dla sposobow prowadzenia obliczen"""

    sekwencyjne = ""
    rownolegle_procesy = "processes"
    rownolegle_watki = "threads"


def podziel(liczby: t.Iterable[int], wielkosc: int) -> list[list[int]]:
    """podziel liste liczb na bloki o wybranej dlugosci"""

    lista_liczb: t.Final[list[int]] = list(liczby)

    return [
        lista_liczb[indeks:indeks + wielkosc]
        for indeks in range(0, len(lista_liczb), wielkosc)
    ]


def zmierz_czas(funkcja: t.Callable[[], t.Any]) -> float:
    """zmierz czas wykonywania funkcji"""

    start: t.Final[float] = tm.time()
    funkcja()
    koniec: t.Final[float] = tm.time()

    return koniec - start


def fibonacci_suma(indeksy: t.Iterable[int]) -> int:
    """oblicz sume liczb ciagu fibonacciego o podanych indeksach w celowo nieoptymalny sposob"""

    suma: int = 0

    for indeks in indeksy:
        lewo: int = 0
        prawo: int = 1

        for _ in range(2, indeks):
            lewo, prawo = prawo, lewo + prawo

        suma += lewo

    return suma


def badanie_szybkosci(
    dlugosci_list: t.Iterable[int],
    podejscie: Podejscie,
    wielkosc_blokow: int,
    ilosc_procesow: int,
) -> list[float]:
    """przetestuj szybkosc obliczania sumy liczb ciagu fibonacciego"""

    czasy: t.Final[list[float]] = []

    for dlugosc in dlugosci_list:
        liczby: list[int] = [16384] * dlugosc
        bloki_liczb: list[list[int]] = podziel(liczby, wielkosc_blokow)

        if podejscie == Podejscie.sekwencyjne:
            czasy.append(zmierz_czas(
                lambda: tuple(map(fibonacci_suma, bloki_liczb)),
            ))
        else:
            czasy.append(zmierz_czas(
                lambda: jl.Parallel(n_jobs = ilosc_procesow, prefer = podejscie)(
                    map(jl.delayed(fibonacci_suma), bloki_liczb),
                ),
            ))

        print(
            f"podejscie: {podejscie.name:>18}, bloki: {wielkosc_blokow:2}, "
            f"procesy: {ilosc_procesow:2}, dlugosc: {dlugosc:3}, czas: {czasy[-1]:4f}"
        )

    return czasy


def glowna() -> None:
    """punkt wyjscia programu"""

    wykres_czasu: t.Final[t.Callable[..., t.Any]] = ft.partial(
        plt.subplot, 2, 2, xlabel = "ilosc iteracji", ylabel = "czas wykonania",
    )
    dlugosci_listy: t.Final[range] = range(64, 64 * 11, 64)

    wykr: plt.Axes = wykres_czasu(1, title = "porownanie podejsc")

    for podejscie in Podejscie:
        czasy: list[float] = badanie_szybkosci(dlugosci_listy, podejscie, 4, -1)

        wykr.plot(dlugosci_listy, czasy, label = podejscie.name)
        wykr.legend()

    wykr = wykres_czasu(2, title = "porownanie wielkosci blokow")

    for wielkosc_blokow in 2, 4, 8, 16, 32, 64:
        czasy = badanie_szybkosci(dlugosci_listy, Podejscie.rownolegle_procesy, wielkosc_blokow, -1)

        wykr.plot(dlugosci_listy, czasy, label = wielkosc_blokow)
        wykr.legend()

    for indeks, podejscie in enumerate((Podejscie.rownolegle_procesy, Podejscie.rownolegle_watki)):
        wykr = wykres_czasu(3 + indeks, title = f"porownanie ilosci procesow ({podejscie.name})")

        for ilosc_procesow in 2, 4, 8:
            czasy = badanie_szybkosci(dlugosci_listy, podejscie, 4, ilosc_procesow)

            wykr.plot(dlugosci_listy, czasy, label = ilosc_procesow)
            wykr.legend()

    plt.show()


if __name__ == "__main__":
    glowna()
