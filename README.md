# Gleichungs-Generator für lineare Gleichungen

Ein Python-basierter Generator für lineare Gleichungen mit einer Unbekannten **x**, der automatisch Arbeits- und Lösungsblätter im DOCX-Format erstellt.

## Features

- **5 Schwierigkeitsgrade** mit steigendem Niveau
- **Lösungen bevorzugt als Brüche** (keine Dezimalzahlen)
- **Durchgehende Nummerierung** (1-23)
- **Visual Complexity Modi**: "clean" oder "mixed" (implizites Mal)
- **Numerische Limits**: Alle Zahlen ≤ 120
- **Duplikaterkennung** über kanonische Formen
- **Schritt-für-Schritt Lösungswege** mit GCD-Normalisierung
- **Definitionsmengen** bei Bruchgleichungen

## Installation

Voraussetzung ist Python 3.10 oder neuer.

### Windows 11 (PowerShell)

1. PowerShell öffnen und in das Projektverzeichnis wechseln.
2. Virtuelle Umgebung erstellen und aktivieren:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   Falls die Aktivierung wegen der Ausführungsrichtlinien scheitert:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
3. Abhängigkeiten installieren:
   ```powershell
   pip install -r requirements.txt
   ```
4. Generator starten:
   ```powershell
   python .\gleichungs_generator.py
   ```

### Linux / macOS (Bash)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python gleichungs_generator.py
```

## Verwendung

### Standardaufruf
```bash
python gleichungs_generator.py
```
Erstellt `arbeitsblatt.docx` und `loesungsblatt.docx` mit Standardeinstellungen.

### Mit Optionen
```bash
# Anderer Zufallssamen für neue Aufgaben
python gleichungs_generator.py --seed 42

# Clean Visual Mode (mit Mal-Zeichen)
python gleichungs_generator.py --visual clean

# Mixed Visual Mode (implizites Mal, Standard)
python gleichungs_generator.py --visual mixed
```

## Schwierigkeitslevel

### Level 1: Einfache lineare Gleichungen (4 Aufgaben)
- x nur auf einer Seite
- Beispiele: `3x - 7 = 12`, `5 - 2x = 17`, `23 = 7 + 2x`

### Level 2: x auf beiden Seiten (4 Aufgaben)  
- Beispiele: `26 - 4x = 5 + 2x`, `3x - 7 = 10x - 25`

### Level 3: Eine Klammer (6 Aufgaben)
- Je 2× mit + vor der Klammer: `4 + (2x - 4) = 4x + 7`
- Je 2× mit - vor der Klammer: `3x - 3 = 12 - (x - 7)`
- Je 2× mit × vor der Klammer: `13(2x - 7) = 12 - 4x`

### Level 4: Gemischte Klammern (4 Aufgaben)
- Mehrere Klammern mit verschiedenen Operationen
- Beispiel: `13(2x - 7) = 12 - (x - 7) + (2x - 4)`

### Level 5: Bruchgleichungen (5 Aufgaben)
- Von einfach bis komplex
- Beispiele: `(1/2)x = 5` bis `(3x + 2)/(x - 1) = 5/3`

**Gesamt: 23 Aufgaben**

## Konfiguration

Die Standardkonfiguration in `gleichungs_generator.py`:

```python
DEFAULT_CONFIG = {
    "seed": 12345,                    # Zufallssamen
    "counts": {                        # Anzahl pro Level
        "L1": 4, "L2": 4, "L3": 6, 
        "L4": 4, "L5": 5
    },
    "coeff_range": (-12, 12),         # Koeffizientenbereich
    "denom_range": (2, 12),            # Nennerbereich
    "prefer_fraction": 0.85,           # 85% Bruchlösungen
    "visual_complexity": "mixed",      # "clean" oder "mixed"
    "solutions": {
        "improper_and_mixed": True     # Zeige beide Bruchformen
    }
}
```

## Numerische Limits

- Alle sichtbaren Zahlen (Zähler, Nenner, Koeffizienten) ≤ 120
- Hauptnenner (kgV) ≤ 120
- GCD-Normalisierung nach Vereinfachung
- Automatisches Resampling bei Überschreitung (max 200 Versuche)

## Tests

```bash
# Alle Tests ausführen
pytest

# Einzelne Tests
pytest test_numeric_limits.py
pytest test_l1_solution.py

# Code-Qualität prüfen
python -m ruff check .
python -m ruff format --check .
```

## Dateien

- `gleichungs_generator.py` - Hauptprogramm
- `arbeitsblatt.docx` - Generierte Aufgaben
- `loesungsblatt.docx` - Lösungen mit Rechenweg
- `requirements.txt` - Python-Abhängigkeiten
- `test_*.py` - Unit-Tests

## Lizenz

Veröffentlicht unter der MIT-Lizenz.
