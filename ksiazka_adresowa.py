#!/usr/bin/env python3

# pylint: disable = invalid-name
# pylint: disable = no-self-argument
# pylint: disable = too-few-public-methods
# pylint: disable = bad-dunder-name
# pylint: disable = magic-value-comparison
# pylint: disable = too-many-try-statements
# pylint: disable = while-used
# pylint: disable = magic-value-comparison
# pylint: disable = redefined-outer-name
# pylint: disable = deprecated-typing-alias

# mypy: warn-unused-configs
# mypy: warn-unreachable
# mypy: warn-return-any
# mypy: warn-redundant-casts
# mypy: warn-unused-ignores
# mypy: warn-return-any
# mypy: disallow-any-generics
# mypy: disallow-subclassing-any
# mypy: disallow-untyped-calls
# mypy: disallow-untyped-defs
# mypy: check-untyped-defs
# mypy: disallow-untyped-decorators
# mypy: disallow-incomplete-defs
# mypy: disallow-any-unimported
# mypy: disallow-any-decorated
# mypy: enable-error-code="redundant-expr"
# mypy: enable-error-code="possibly-undefined"
# mypy: enable-error-code="truthy-bool"
# mypy: enable-error-code="truthy-iterable"
# mypy: enable-error-code="ignore-without-code"
# mypy: enable-error-code="unused-awaitable"
# mypy: show-error-context
# mypy: show-error-end
# mypy: local-partial-types
# mypy: no-implicit-reexport
# mypy: strict-equality
# mypy: extra-checks
# mypy: pretty

"""program do zarzadzania ksiazka adresowa w telefonie komorkowym"""

from __future__ import annotations

import datetime as dt
import enum as en
import inspect as ins
import operator as op
import typing as t
import pathlib as pl
import random as rn
import readline  # pylint: disable = unused-import
import shlex as sh
import sys
import textwrap as tw

import attrs as at
import phonenumbers as pn
import sortedcontainers as sc
import sqlite_utils as su
import tabulate as tb
import pytest as pt


class Kierunek(en.StrEnum):
    """wyliczenie kierunku rozmowy uzywane zamiast parametrow tekstowych"""

    p = "przychodzaca"
    w = "wychodzaca"


class Rozmowa(t.NamedTuple):
    """krotka z parametrami rozmowy telefonicznej z kontaktem"""

    data: dt.datetime
    dlugosc: dt.timedelta
    kierunek: Kierunek


@at.mutable
class Kontakt:
    """klasa przechowujaca numer telefonu i liste rozmow posortowana zgodnie z data"""

    numer: pn.PhoneNumber
    rozmowy: sc.SortedList[Rozmowa] = at.field(
        factory = lambda: sc.SortedList(key = op.attrgetter("data")),
        init = False,
    )

    def ladny_numer(to: t.Self) -> str:
        """utworz reprezentacje numeru z odstepami pomiedzy grupami cyfr"""

        return pn.format_number(
            numobj = to.numer,
            num_format = pn.PhoneNumberFormat.INTERNATIONAL,
        )


@at.frozen
class Sql:
    """klasa z metodami statycznymi do interakcji z baza danych sqlite"""

    @staticmethod
    def z(baza: su.Database) -> Ksiazka:
        """wczytaj dane z tabel bazy danych i utworz ksiazke adresowa"""

        ksiazka: Ksiazka = Ksiazka()

        for rzad in baza["kontakty"].rows:
            ksiazka.kontakty[rzad["nazwa"]] = Kontakt(pn.parse(rzad["numer"]))

        for rzad in baza["rozmowy"].rows:
            ksiazka.kontakty[rzad["nazwa"]].rozmowy.add(
                Rozmowa(
                    data = dt.datetime.fromtimestamp(rzad["data"]),
                    dlugosc = dt.timedelta(seconds = rzad["dlugosc"]),
                    kierunek = Kierunek[rzad["kierunek"]],
                ),
            )

        return ksiazka

    @staticmethod
    def do(baza: su.Database, ksiazka: Ksiazka) -> None:
        """utworz w bazie nowe tabele i przenies do nich zawartosc ksiazki"""

        for nazwa_tabeli in ("kontakty", "rozmowy"):
            baza[nazwa_tabeli].drop(ignore = True)

        for nazwa, kontakt in ksiazka.kontakty.items():
            # kolumny maja automatycznie ustawiony poprawny typ danych,
            # co mozna zobaczyc po wpisaniu `sqlite3 ksiazka.db .dump`
            baza["kontakty"].insert(  # type: ignore [union-attr]
                {
                    "nazwa": nazwa,
                    "numer": kontakt.ladny_numer(),
                },
            )

            for data, dlugosc, kierunek in kontakt.rozmowy:
                baza["rozmowy"].insert(  # type: ignore [union-attr]
                    {
                        "nazwa": nazwa,
                        "data": data.timestamp(),
                        "dlugosc": dlugosc.total_seconds(),
                        "kierunek": kierunek.name,
                    },
                )

@at.frozen
class Tabela:
    """klasa umozliwiajaca wyswietlenie tabeli"""

    tytul: t.Sequence[str] = ()
    kolumny: t.Sequence[str] = ()
    rzedy: t.Sequence[t.Sequence[str]] = ()

    def zlacz(to: t.Self, styl: str = "rounded_outline") -> str:
        """utworz tabele tytulowa oraz tabele ze wskazanymi kolumnami i rzedami"""

        return "\n".join(
            (
                tb.tabulate(
                    tabular_data = (to.tytul,),
                    tablefmt = styl,
                ),
                tb.tabulate(
                    tabular_data = to.rzedy,
                    headers = to.kolumny,
                    tablefmt = styl,
                ),
            ),
        )


class Szablon(en.StrEnum):
    """wyliczenie szablonow dla wyswietlanych komunikatow"""

    pogadano = "porozmawiano z kontaktem {}"
    usunieto = "usunieto kontakt {}"
    usunieto_wszystko = "wyczyszczono ksiazke adresowa"
    dodano = "dodano kontakt {}"
    zmieniono_nazwe = "zmieniono nazwe kontaktu {} na {}"
    zmieniono_numer = "zmieniono numer kontaktu {} na {}"
    zle_polecenie = (
        "polecenie jest nieprawidlowe; komenda pomoc wyswietla instrukcje"
    )
    zla_nazwa = "nie znaleziono kontaktu o podanej nazwie"
    jest_nazwa = "kontakt z podana nazwa jest juz obecny"
    zly_numer = (
        "numer powinien skladac sie z cyfr i byc poprzedzony kodem krajowym "
        "ze znakiem \"+\""
    )
    dzien_dobry = "polaczono z baza danych"
    do_widzenia = "do widzenia"


@at.frozen
class Komunikat:
    """klasa umozliwiajaca wyswietlenie komunikatu tekstowego"""

    szablon: Szablon
    elementy: tuple[str, ...] = ()

    def zlacz(to: t.Self) -> str:
        """utworz pelny komunikat"""

        return to.szablon.format(*to.elementy)


@at.mutable
class Ksiazka:
    """klasa przechowujaca liste kontaktow oraz metody do zarzadzania nia"""

    baza: su.Database | None = None
    kontakty: sc.SortedDict[str, Kontakt] = at.field(
        factory = lambda: sc.SortedDict(str.casefold),
        init = False,
    )

    def __attrs_post_init__(to: t.Self) -> None:
        """wczytaj stan z bazy danych"""

        if to.baza:
            to.kontakty = Sql.z(to.baza).kontakty

    def __enter__(to: t.Self) -> t.Self:
        """inicjalizuj menadzer kontekstu"""

        return to

    def __exit__(to: t.Self, *_: t.Any) -> None:
        """zakoncz menadzer kontekstu"""

        to.zapisz()

    def zapisz(to: t.Self) -> None:
        """zapisz stan do bazy danych"""

        if to.baza:
            return Sql.do(to.baza, to)

        raise AttributeError

    def lista(to: t.Self) -> Tabela:
        """wyswietl zawartosc ksiazki"""

        return Tabela(
            tytul = ("lista kontaktow", str(len(to.kontakty))),
            kolumny = ("nazwa", "numer"),
            rzedy = [
                (nazwa, kontakt.ladny_numer())
                for nazwa, kontakt in to.kontakty.items()
            ],
        )

    def pogadaj(to: t.Self, nazwa: str) -> Komunikat:
        """dodaj rozmowe o losowej dlugosci"""

        to.kontakty[nazwa].rozmowy.add(
            Rozmowa(
                data = dt.datetime.now(),
                dlugosc = dt.timedelta(seconds = rn.randrange(60, 600)),
                kierunek = Kierunek.w,
            ),
        )

        return Komunikat(Szablon.pogadano, (nazwa,))

    def info(to: t.Self, nazwa: str) -> Tabela:
        """wyswietl szczegoly kontaktu"""

        kontakt: Kontakt = to.kontakty[nazwa]

        return Tabela(
            tytul = (nazwa, kontakt.ladny_numer()),
            kolumny = ("data", "dlugosc", "kierunek"),
            rzedy = [
                (
                    str(data).split(".", maxsplit = 1)[0],
                    str(dlugosc).split(".", maxsplit = 1)[0],
                    kierunek,
                )
                for data, dlugosc, kierunek in reversed(kontakt.rozmowy)
            ],
        )

    def usun(to: t.Self, nazwa: str) -> Komunikat:
        """usun kontakt; \"*\" usuwa wszystko"""

        if nazwa == "*":
            to.kontakty.clear()

            return Komunikat(Szablon.usunieto_wszystko)

        to.kontakty.pop(nazwa)

        return Komunikat(Szablon.usunieto, (nazwa,))

    def poprawna(to: t.Self, nowa_nazwa: str) -> None:
        """upewnij sie ze nowa nazwa kontaktu jest odpowiednia"""

        if nowa_nazwa == "*":
            raise ValueError

        if nowa_nazwa in to.kontakty:
            raise RuntimeError

    def dodaj(to: t.Self, nowa_nazwa: str, nowy_numer: str) -> Komunikat:
        """dodaj nowy kontakt"""

        to.poprawna(nowa_nazwa)
        to.kontakty[nowa_nazwa] = Kontakt(pn.parse(nowy_numer))

        return Komunikat(Szablon.dodano, (nowa_nazwa,))

    def zmiennazwe(to: t.Self, nazwa: str, nowa_nazwa: str) -> Komunikat:
        """zmien nazwe kontaktu"""

        to.poprawna(nowa_nazwa)
        to.kontakty[nowa_nazwa] = to.kontakty.pop(nazwa)

        return Komunikat(Szablon.zmieniono_nazwe, (nazwa, nowa_nazwa))

    def zmiennumer(to: t.Self, nazwa: str, nowy_numer: str) -> Komunikat:
        """zmien numer kontaktu"""

        to.kontakty[nazwa].numer = pn.parse(nowy_numer)

        return Komunikat(Szablon.zmieniono_numer, (nazwa, nowy_numer))


@at.mutable
class KsiazkaIU:
    """klasa zarzadzajaca ksiazka adresowa za pomoca interfejsu uzytkownika"""

    ksiazka: Ksiazka
    komendy: dict[str, t.Callable[..., Tabela | Komunikat]] = at.field(
        init = False,
    )

    def __attrs_post_init__(to: t.Self) -> None:
        """ustaw komendy dostepne w interfejsie uzytkownika"""

        komendy_interfejsu: tuple[t.Callable[..., Tabela | Komunikat], ...] = (
            to.ksiazka.lista,
            to.ksiazka.pogadaj,
            to.ksiazka.info,
            to.ksiazka.usun,
            to.ksiazka.dodaj,
            to.ksiazka.zmiennazwe,
            to.ksiazka.zmiennumer,
            to.pomoc,
        )
        to.komendy = {
            funkcja.__name__: funkcja
            for funkcja in komendy_interfejsu
        }

    def pomoc(to: t.Self) -> Tabela:
        """wyswietl instrukcje"""

        info: list[tuple[str, str]] = [
            ("", ""),
            ("*nazwy lub numery zawierajace spacje", ""),
            ("powinny byc otoczone cudzyslowami", ""),
        ]

        return Tabela(
            tytul = ("dostepne komendy", "wersja 2"),
            kolumny = ("komenda", "opis"),
            rzedy = [
                (parametry(funkcja), funkcja.__doc__ or "")
                for funkcja in to.komendy.values()
            ] + info,
        )

    def polecenie(to: t.Self, tresc: str) -> Tabela | Komunikat:
        """zinterpretuj i wykonaj polecenie uzytkownika"""

        try:
            argumenty: list[str] = [
                argument.strip()
                for argument in sh.split(tresc)
            ]

            return to.komendy.get(argumenty[0])(*argumenty[1:])  # type: ignore [misc]
        except (AttributeError, IndexError, TypeError, ValueError):
            return Komunikat(Szablon.zle_polecenie)
        except KeyError:
            return Komunikat(Szablon.zla_nazwa)
        except RuntimeError:
            return Komunikat(Szablon.jest_nazwa)
        except pn.phonenumberutil.NumberParseException:
            return Komunikat(Szablon.zly_numer)

    def otworz(to: t.Self) -> t.NoReturn:
        """otworz interfejs uzytkownika"""

        if to.ksiazka.baza:
            print(Szablon.dzien_dobry)

        while True:
            try:
                wyjscie: Tabela | Komunikat = to.polecenie(input("\n> "))
            except (EOFError, KeyboardInterrupt):
                print("\n" + Szablon.do_widzenia)
                sys.exit()

            print(wyjscie.zlacz())


def parametry(metoda: t.Callable[..., t.Any]) -> str:
    """polacz nazwe metody z nazwami jej parametrow"""

    return metoda.__name__ + "".join(
        f" [{parametr}]"
        for parametr in ins.getfullargspec(metoda).args[1:]
    )


def bez_wciec(tekst: str) -> str:
    """usun wciecia z wieloliniowego ciagu tekstu"""

    return tw.dedent(tekst).strip()


def glowna() -> t.NoReturn:  # pragma: no cover
    """punkt wyjscia programu"""

    # try:
    #     ksiazka: Ksiazka = Ksiazka(su.Database("ksiazka.db"))
    #     KsiazkaIU(ksiazka).otworz()
    # finally:
    #     ksiazka.zapisz()

    with Ksiazka(su.Database("ksiazka.db")) as ksiazka:
        KsiazkaIU(ksiazka).otworz()


# pip install mypy pytest-mypy
# statyczne typowanie obecne w kodzie pozwala uniknac pisania testow sprawdzajacych typy
# dzieki czemu ich ilosc moze byc duzo mniejsza
pytest_plugins: tuple[str] = ("mypy",)


def test_kontakt_ladny_numer() -> None:
    """przetestuj formatowanie numeru"""

    kontakt: Kontakt = Kontakt(pn.parse("+48123123123"))

    assert kontakt.ladny_numer() == "+48 12 312 31 23"


@pt.fixture
def baza_testowa() -> t.Generator[su.Database, None, None]:
    """utworz baze danych uzywana do testow a nastepnie posprzataj po sobie"""

    sciezka: pl.Path = pl.Path("ksiazka_test.db")

    assert not sciezka.exists()

    yield su.Database(sciezka)

    sciezka.unlink()


@pt.fixture
def ksiazka_testowa() -> Ksiazka:
    """utworz ksiazke adresowa uzywana do testow"""

    ksiazka: Ksiazka = Ksiazka()

    ksiazka.kontakty["Bolek"] = Kontakt(pn.parse("+48 12 312 31 23"))
    ksiazka.kontakty["Lolek"] = Kontakt(pn.parse("+48 32 132 13 21"))
    ksiazka.kontakty["Lolek"].rozmowy.update(
        (
            Rozmowa(
                data = dt.datetime(2024, 1, 20, 12, 30, 10, 102030),
                dlugosc = dt.timedelta(0, 123, 456789),
                kierunek = Kierunek.p,
            ),
            Rozmowa(
                data = dt.datetime(2024, 1, 20, 14, 58, 0, 999999),
                dlugosc = dt.timedelta(0, 423, 1),
                kierunek = Kierunek.w,
            ),
        ),
    )

    return ksiazka


zapytanie: str = bez_wciec(
    """
        BEGIN TRANSACTION;
        CREATE TABLE [kontakty] (
           [nazwa] TEXT,
           [numer] TEXT
        );
        INSERT INTO "kontakty" VALUES('Bolek','+48 12 312 31 23');
        INSERT INTO "kontakty" VALUES('Lolek','+48 32 132 13 21');
        CREATE TABLE [rozmowy] (
           [nazwa] TEXT,
           [data] FLOAT,
           [dlugosc] FLOAT,
           [kierunek] TEXT
        );
        INSERT INTO "rozmowy" VALUES('Lolek',1705750210.10203,123.456789,'p');
        INSERT INTO "rozmowy" VALUES('Lolek',1.70575908099999904e+09,423.000001,'w');
        COMMIT;
    """,
)


def test_do_bazy(baza_testowa: su.Database, ksiazka_testowa: Ksiazka) -> None:
    """przetestuj przenoszenie danych do bazy"""

    Sql.do(baza_testowa, ksiazka_testowa)

    assert "\n".join(baza_testowa.conn.iterdump()) == zapytanie


def test_z_bazy(baza_testowa: su.Database, ksiazka_testowa: Ksiazka) -> None:
    """przetestuj wczytywanie danych z bazy"""

    baza_testowa.executescript(zapytanie)

    assert Sql.z(baza_testowa) == ksiazka_testowa


def test_tabela() -> None:
    """przetestuj tworzenie tabel"""

    tabela: Tabela = Tabela(
        tytul = ("tytul", "podtytul"),
        kolumny = ("naglowek 1", "naglowek 2"),
        rzedy = (("pole 1", "pole 2"), ("pole 3", "pole 4")),
    )

    assert tabela.zlacz(styl = "rounded_outline") == bez_wciec(
        """
            ╭───────┬──────────╮
            │ tytul │ podtytul │
            ╰───────┴──────────╯
            ╭──────────────┬──────────────╮
            │ naglowek 1   │ naglowek 2   │
            ├──────────────┼──────────────┤
            │ pole 1       │ pole 2       │
            │ pole 3       │ pole 4       │
            ╰──────────────┴──────────────╯
        """,
    )


def test_komunikat() -> None:
    """przetestuj tworzenie komunikatow"""

    nazwa: str = "anon"

    komunikat: Komunikat = Komunikat(Szablon.dodano, (nazwa,))

    assert nazwa in komunikat.zlacz()


def test_zapisz(baza_testowa: su.Database) -> None:
    """przetestuj zapisywanie ksiazki do bazy danych"""

    nazwa: str = "Rumcajs"
    numer: str = "+420 346 343 445"

    with Ksiazka(baza_testowa) as ksiazka:
        ksiazka.kontakty[nazwa] = Kontakt(pn.parse(numer))

    umiesc: str = f"INSERT INTO \"kontakty\" VALUES('{nazwa}','{numer}');"

    assert umiesc in baza_testowa.conn.iterdump()


def test_zapisz_blad() -> None:
    """przetestuj zapisywanie ksiazki bez bazy danych"""

    ksiazka: Ksiazka = Ksiazka()

    with pt.raises(AttributeError):
        ksiazka.zapisz()


def test_lista(ksiazka_testowa: Ksiazka) -> None:
    """przetestuj wyswietlanie listy kontaktow"""

    assert ksiazka_testowa.lista() == Tabela(
        tytul = ("lista kontaktow", "2"),
        kolumny = ("nazwa", "numer"),
        rzedy = [
            ("Bolek", "+48 12 312 31 23"),
            ("Lolek", "+48 32 132 13 21"),
        ],
    )


def test_pogadaj(ksiazka_testowa: Ksiazka) -> None:
    """przetestuj dodawanie rozmowy do kontaktu"""

    nazwa: str = "Bolek"

    komunikat: Komunikat = ksiazka_testowa.pogadaj(nazwa)

    assert komunikat.elementy == (nazwa,)

    rozmowa: Rozmowa = ksiazka_testowa.kontakty[nazwa].rozmowy[-1]

    assert rozmowa.dlugosc > dt.timedelta(0)
    assert rozmowa.data > dt.datetime(1, 1, 1)


def test_info(ksiazka_testowa: Ksiazka) -> None:
    """przetestuj wyswietlanie szczegolow kontaktu"""

    assert ksiazka_testowa.info("Lolek") == Tabela(
        tytul = ("Lolek", "+48 32 132 13 21"),
        kolumny = ("data", "dlugosc", "kierunek"),
        rzedy = [
            ("2024-01-20 14:58:00", "0:07:03", "wychodzaca"),
            ("2024-01-20 12:30:10", "0:02:03", "przychodzaca"),
        ],
    )


def test_usun(ksiazka_testowa: Ksiazka) -> None:
    """przetestuj usuwanie kontaktu z ksiazki"""

    nazwa: str = "Lolek"

    komunikat: Komunikat = ksiazka_testowa.usun(nazwa)

    assert komunikat.elementy == (nazwa,)
    assert nazwa not in ksiazka_testowa.kontakty


def test_usun_wszystko(ksiazka_testowa: Ksiazka) -> None:
    """przetestuj czyszczenie ksiazki"""

    komunikat: Komunikat = ksiazka_testowa.usun("*")

    assert komunikat.elementy == ()
    assert not ksiazka_testowa.kontakty


def test_poprawna(ksiazka_testowa: Ksiazka) -> None:
    """przetestuj ignorowanie poprawnych nazw"""

    ksiazka_testowa.poprawna("ktos")


def test_poprawna_blad(ksiazka_testowa: Ksiazka) -> None:
    """przetestuj wykrywanie niepoprawnych nazw"""

    with pt.raises(ValueError):
        ksiazka_testowa.poprawna("*")

    with pt.raises(RuntimeError):
        ksiazka_testowa.poprawna("Bolek")


def test_dodaj(ksiazka_testowa: Ksiazka) -> None:
    """przetestuj dodawanie kontaktu do ksiazki"""

    nazwa: str = "Krecik"
    numer: str = "+420 321 321 321"

    komunikat: Komunikat = ksiazka_testowa.dodaj(nazwa, numer)

    assert komunikat.elementy == (nazwa,)
    assert nazwa in ksiazka_testowa.kontakty

    kontakt: Kontakt = ksiazka_testowa.kontakty[nazwa]

    assert kontakt == Kontakt(pn.parse(numer))


def test_zmiennazwe(ksiazka_testowa: Ksiazka) -> None:
    """przetestuj zmienianie nazwy kontaktu"""

    nazwa: str = "Lolek"
    nowa_nazwa: str = "Lmaolek"

    komunikat: Komunikat = ksiazka_testowa.zmiennazwe(nazwa, nowa_nazwa)

    assert komunikat.elementy == (nazwa, nowa_nazwa)
    assert nowa_nazwa in ksiazka_testowa.kontakty


def test_zmiennumer(ksiazka_testowa: Ksiazka) -> None:
    """przetestuj zmienianie numeru kontaktu"""

    nazwa: str = "Bolek"
    numer: str = "+48123456789"

    komunikat: Komunikat = ksiazka_testowa.zmiennumer(nazwa, numer)

    assert komunikat.elementy == (nazwa, numer)
    assert ksiazka_testowa.kontakty[nazwa].numer == pn.parse(numer)


@pt.fixture
def interfejs_testowy(ksiazka_testowa: Ksiazka) -> KsiazkaIU:
    """utworz interfejs ksiazki uzywany do testow"""

    return KsiazkaIU(ksiazka_testowa)


def test_interfejsksiazki(interfejs_testowy: KsiazkaIU) -> None:
    """przetestuj tworzenie klasy z interfejsem uzytkownika"""

    assert len(interfejs_testowy.komendy) > 0


def test_pomoc(interfejs_testowy: KsiazkaIU) -> None:
    """przetestuj wyswietlanie instrukcji"""

    assert len(interfejs_testowy.pomoc().rzedy) > 0


def test_polecenie(interfejs_testowy: KsiazkaIU) -> None:
    """przetestuj interpretacje polecenia"""

    interfejs_testowy.polecenie(
        "dodaj \"ktos\" \"+48 111 222 333\"",
    )

    assert "ktos" in interfejs_testowy.ksiazka.kontakty


@pt.mark.parametrize(
    ("z_baza",),
    ((True,), (False,)),
)
def test_konsola(
    z_baza: bool,
    capsys: pt.CaptureFixture[str],
    monkeypatch: pt.MonkeyPatch,
    interfejs_testowy: KsiazkaIU,
    baza_testowa: su.Database,
) -> None:
    """przetestuj symulowane czynnosci uzytkownika interfejsu"""

    nazwa: str = "Bolek"

    def wyjdz() -> t.Generator[str, None, None]:
        """wykonaj jedna iteracje i nastepnie wyjdz z programu za pomoca sygnalu"""

        yield f"pogadaj {nazwa}"

        raise KeyboardInterrupt

    iteracja: t.Generator[str, None, None] = wyjdz()

    monkeypatch.setattr("builtins.input", lambda _: next(iteracja))

    oczekiwane_wyjscie: str = bez_wciec(
        f"""
            {Komunikat(Szablon.pogadano, (nazwa,)).zlacz()}

            {Szablon.do_widzenia}
        """,
    ) + "\n"

    if z_baza:
        interfejs_testowy.ksiazka.baza = baza_testowa
        oczekiwane_wyjscie = f"{Szablon.dzien_dobry}\n{oczekiwane_wyjscie}"

    try:
        interfejs_testowy.otworz()
    except SystemExit:
        assert capsys.readouterr().out == oczekiwane_wyjscie


@pt.mark.parametrize(
    ("polecenie", "rezultat"),
    (
        ("b", Szablon.zle_polecenie),
        ("zmiennazwe ktos \"ktos inny\"", Szablon.zla_nazwa),
        ("zmiennazwe Lolek Bolek", Szablon.jest_nazwa),
        ("zmiennumer Lolek +48", Szablon.zly_numer),
    ),
)
def test_polecenie_blad(
    polecenie: str,
    rezultat: Szablon,
    interfejs_testowy: KsiazkaIU,
) -> None:
    """przetestuj bledne polecenia"""

    wyjscie: Tabela | Komunikat = interfejs_testowy.polecenie(polecenie)

    assert isinstance(wyjscie, Komunikat)
    assert wyjscie.szablon == rezultat


def test_parametry() -> None:
    """przetestuj laczenie nazwy i parametrow metody"""

    assert parametry(Tabela.zlacz) == "zlacz [styl]"


def test_bez_wciec() -> None:
    """przetestuj usuwanie wciec z ciagu tekstu"""

    tekst: str = """
        h
        a
        l
        o
    """

    assert bez_wciec(tekst) == "h\na\nl\no"


# pip install coverage
# program ten mowi ze w powyzszych testach zostalo osiagniete 100-procentowe pokrycie kodu


if __name__ == "__main__":  # pragma: no cover
    glowna()
