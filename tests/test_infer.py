"""Antwort-Schnitt: was ``generate`` liefert, ist nicht immer ein Tensor.

Der Fund auf einer echten Colab-T4: ``Lfm3OmniForConditionalGeneration.generate``
gibt ein ``Lfm3OmniGenerateOutput`` zurück. Der alte Code indizierte es wie einen
Tensor, las eine leere Sequenz und meldete 0 % Trefferquote – beim Basismodell
*und* beim Adapter, obwohl das Modell im nächsten-Token-Test korrekt „Hallo"
vorhersagte.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from lfm3ha.infer import extract_new_tokens


class FakeTensor:
    """Gerade genug Tensor, um die Schnittlogik zu prüfen."""

    def __init__(self, rows: list[list[int]]):
        self.rows = rows
        self.shape = (len(rows), len(rows[0]))

    def __getitem__(self, index):
        return self.rows[index]


def test_omni_ausgabeobjekt_wird_richtig_geschnitten():
    """Prompt 3 Token, Antwort 2 Token, danach Padding."""
    result = SimpleNamespace(
        sequences=FakeTensor([[10, 11, 12, 13, 14, 15, 15]]),
        sequence_lengths=[5],  # 5 = 3 Prompt + 2 Antwort
    )
    assert extract_new_tokens(result, prompt_len=3) == [13, 14]


def test_reiner_tensor_funktioniert_weiter():
    assert extract_new_tokens(FakeTensor([[10, 11, 12, 13]]), prompt_len=2) == [12, 13]


def test_objekt_ohne_lengths_faellt_auf_die_volle_laenge_zurueck():
    result = SimpleNamespace(sequences=FakeTensor([[10, 11, 12, 13]]))
    assert extract_new_tokens(result, prompt_len=2) == [12, 13]


def test_liste_von_sequenzen():
    result = SimpleNamespace(sequences=[FakeTensor([[10, 11, 12]])], sequence_lengths=[3])
    assert extract_new_tokens(result, prompt_len=1) == [11, 12]


def test_leere_antwort_bleibt_leer():
    result = SimpleNamespace(sequences=FakeTensor([[10, 11]]), sequence_lengths=[2])
    assert extract_new_tokens(result, prompt_len=2) == []


def test_prompt_laenger_als_lengths_wirft_nicht():
    """Kann passieren, wenn ein Modell die Prompt-Länge nicht mitzählt."""
    result = SimpleNamespace(sequences=FakeTensor([[10, 11, 12]]), sequence_lengths=[1])
    assert extract_new_tokens(result, prompt_len=3) == []


@pytest.mark.parametrize("prompt_len", [0, 1, 2])
def test_schnitt_passt_zur_promptlaenge(prompt_len):
    result = SimpleNamespace(
        sequences=FakeTensor([[1, 2, 3, 4, 5]]), sequence_lengths=[5]
    )
    assert extract_new_tokens(result, prompt_len) == [1, 2, 3, 4, 5][prompt_len:]
