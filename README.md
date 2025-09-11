## Installation unter Windows 11 (PowerShell)

1. PowerShell öffnen und in das Projektverzeichnis wechseln.
2. Virtuelle Umgebung erstellen und aktivieren:
   ```powershell
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
   Falls die Aktivierung wegen der Ausführungsrichtlinien scheitert, führe einmal:
   ```powershell
   Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
   ```
   aus und versuche es erneut.
3. Abhängigkeiten installieren (falls eine `requirements.txt` vorhanden ist):
   ```powershell
   pip install -r requirements.txt
   ```
4. Generator starten:
   ```powershell
   python .\gleichungsgenerator.py
   ```

---

Ich brauche für meine Teilnehmer soetwas wie einen "gleichungs-generator".

Nur eine unbekannte und zwar "x".

Am Anfang die ganz leichten Beispiele wo die Anordnung z.B: so ist:

3x-7=12.   Oder:
5-2x=17.    Oder:
23=7+2x.   Usw.

Dann die leichten Beispiele die so gestaltet sind:

26-4x=5+2x.   Oder:
3x-7=10x-25.  Usw.

Dann mittlere Beispiele die eine Klammer haben, entweder plus vor der klammer, oder ein minus vor der klammer, oder ein Mal vor einer Klammer, das aber nicht immer sichtbar sein muss! z.B.:

4+(2x-4)=4x+7...(plus vor der Klammer)
3x-3=12-(x-7)...(Minus vor der klammer)
13(2x-7)=12-4x...(Multiplikation vor der klammer)

Dann ein bisschen schwerere Beispiele, wo diese Klammerarten gemischt werden:
13(2x-7)=12-(x-7)+(2x-4)

Und zum schluss auch welche mit Brüchen:

Und zwar von ½x=5

Bis zu {3(4x-2)}/{-2(3x-12)}={(7x-2)3}/{(-6)(2x-3)}

Das letzte Beispiel stellt eher schon hohen Schwierigkeitsgrad dar.

Es sollte darauf geachtet werden, dass das Ergebnis moglichst oft in einer Bruchdarstellung endet, das gilt fur alle Beispiele (am besten ware, dass es mit einer Division sehr schwer ist die Lösung zu finden, aber man es in einem bruch sehr einfach lösen und darstellen kann (Ganze und Bruchteile)

Ich brauche sowas in Python das Ding soll mir Beispiele generieren. Sie in ein docx Dokument schreiben, genug abstand zwischen den aufgaben lassen, damit man Platz zum lösen hat. Die Gleichungen sollen immer losbar sein. Ich will eine zweite Ausgabe, quasi ein Lösungsblatt mit Rechenweg.



Hier ist die **vollständige, strukturierte und präzise Spezifikation** deines gewünschten **Gleichungs-Generators** für lineare Gleichungen mit einer Unbekannten $x$.

---

## 📘 **Projekt: Gleichungs-Generator mit Lösungen (Python-basiert)**

---

### 🎯 **Zielsetzung**

Ein automatischer Generator soll lineare Gleichungen der Form

$$
A(x) = B(x)
$$

erzeugen, die **eine Unbekannte $x$** enthalten und in **5 Schwierigkeitslevel** unterteilt sind.

#### Eigenschaften aller Gleichungen:

* Nur **eine Variable: $x$**
* Alle Gleichungen sind **eindeutig lösbar**
* Die **Lösung soll möglichst oft ein Bruch** sein (z. B. $x = \frac{5}{3}$ statt $x = 2$)
* Die **Gleichungen sollen optisch unübersichtlich**, aber **rechnerisch korrekt und lösbar** sein
* Die Lösungen sollen als **Brüche (auch gemischt, z. B. $1 \frac{3}{4}$)** dargestellt werden
* Die **Darstellung als Dezimalzahl ist unerwünscht**
* **Divisionen sollten visuell kompliziert**, aber algebraisch einfach auflösbar sein

---

### 📚 **Struktur des Arbeitsblatts**

| Level | Beschreibung                                                    | Anzahl Aufgaben |
| ----- | --------------------------------------------------------------- | --------------- |
| 1     | Einfache lineare Gleichungen mit x nur auf einer Seite          | 4               |
| 2     | Lineare Gleichungen mit x auf beiden Seiten                     | 4               |
| 3     | Gleichungen mit Klammerausdrücken (jeweils 2× +, –, × gemischt) | 6               |
| 4     | Gemischte Klammerarten (mehrgliedrig, verschachtelt, verteilt)  | 4               |
| 5     | Bruchgleichungen, kontinuierlich im Schwierigkeitsgrad steigend | 5               |

---

## 🔢 **Details zu den Schwierigkeitsgraden**

---

### 🟩 **Level 1: Einfache lineare Gleichungen (x auf einer Seite)**

**Struktur:**

* Terme wie $ax + b = c$, oder $a - bx = c$, oder auch in umgestellter Form

**Beispielhafte Formen:**

* $3x - 7 = 12$
* $5 - 2x = 17$
* $23 = 7 + 2x$
* $-4x + 9 = 1$

**Ziel:** Einführung in das Umformen linearer Gleichungen.

---

### 🟨 **Level 2: x auf beiden Seiten**

**Struktur:**

* Gleichungen wie $ax + b = dx + e$, mit x auf beiden Seiten

**Beispielhafte Formen:**

* $26 - 4x = 5 + 2x$
* $3x - 7 = 10x - 25$
* $-2x + 11 = 3x - 9$
* $14x + 3 = 8x + 21$

**Ziel:** Erweiterung des Umformens durch Variablen-Ausgleich.

---

### 🟧 **Level 3: Gleichungen mit einer Klammer**

**Aufteilung:**

* 2 Gleichungen mit **+ vor der Klammer**
* 2 Gleichungen mit **– vor der Klammer**
* 2 Gleichungen mit **× vor der Klammer (Multiplikation)**
  → Malzeichen kann **implizit** (weggelassen) sein

**Beispielhafte Formen:**

* $4 + (2x - 4) = 4x + 7$     (+ vor der Klammer)
* $3x - 3 = 12 - (x - 7)$     (– vor der Klammer)
* $13(2x - 7) = 12 - 4x$      (× vor der Klammer)

**Ziel:** Verständnis für Klammerauflösung in verschiedenen Kontexten.

---

### 🟨 **Level 4: Gemischte Klammerarten in einer Gleichung**

**Struktur:**

* Kombination aus +, –, × vor Klammern in **einer Gleichung**
* Kann auf beiden Seiten Klammern enthalten
* Eventuell verschachtelte oder mehrfach vorkommende Klammern

**Beispielhafte Formen:**

* $13(2x - 7) = 12 - (x - 7) + (2x - 4)$
* $2x + (3x - 5) = 4(2x - 1) - (x + 2)$
* $(4x - 5) - 3(2x + 1) = (x + 3) - 2$
* $7(x - 3) - (2x + 1) = 3(x - 4) + 6$

**Ziel:** Umgang mit strukturell komplexeren Gleichungen.

---

### 🟥 **Level 5: Bruchgleichungen (kontinuierlich schwieriger)**

**Aufbau:**

* Von **einfachen Bruchformen** bis zu **verschachtelten, unübersichtlichen Ausdrücken**
* Lösungen sind absichtlich als **Brüche darstellbar**
* Ziel ist es, Aufgaben zu erzeugen, deren Lösung **in Bruchdarstellung leichter** als per Division zu finden ist

**Beispielhafte Formen:**

1. $\frac{1}{2}x = 5$
2. $\frac{2x + 3}{4} = 7$
3. $\frac{2x - 3}{x + 1} = 1$
4. $\frac{3(4x - 2)}{-2(3x - 12)} = \frac{(7x - 2) \cdot 3}{(-6)(2x - 3)}$
5. $\frac{5x - 4}{2} + \frac{3x + 1}{3} = \frac{4x + 7}{6}$

**Ziel:** Verinnerlichung des Umgangs mit Gleichungen mit Brüchen, inkl. Hauptnennerbildung, Klammerverarbeitung und Termvereinfachung.

---

## 📝 **Dokumentenerstellung (automatisiert durch Python)**

### 📄 **Arbeitsblatt (`arbeitsblatt.docx`)**

* Alle **23 Gleichungen** (nach Leveln gruppiert)
* Jede Aufgabe mit **genügend Leerraum** darunter (Platz zum Rechnen)
* **Levelkennzeichnung** optional am Seitenrand oder als Überschrift
* **Keine Lösungen sichtbar**

---

### 📄 **Lösungsblatt (`loesungsblatt.docx`)**

* **Nummerierte Lösungen** passend zu den Aufgaben
* **Schritt-für-Schritt Rechenweg** zu jeder Aufgabe
* Ergebnisse möglichst als **Bruch** oder **gemischte Zahl**, wenn nötig
* Optional: grafische Formatierung (z. B. Einrückung bei Rechenschritten)

---

## 🧩 Zusammenfassung – Was der Generator können muss:

| Feature             | Beschreibung                                                |
| ------------------- | ----------------------------------------------------------- |
| Gleichungstyp       | Lineare Gleichungen mit einer Variablen $x$                 |
| Schwierigkeitsgrade | 5 Level (von einfach bis bruchbasiert-komplex)              |
| Lösungseigenschaft  | Immer eindeutig lösbar, Ergebnis oft Bruch                  |
| Format              | Word-Dokumente: Arbeitsblatt + Lösungsblatt                 |
| Zusatz              | Platz zum Rechnen, verständlicher Rechenweg im Lösungsblatt |
| Anzahl Aufgaben     | Genau 23 (verteilt nach Anweisung)                          |
| Bruchdarstellung    | Priorität bei Lösungsausgabe, keine Dezimalergebnisse       |
| Komplexität         | Optisch komplex, rechnerisch logisch aufgebaut              |

---

---

### 🎯 **Zielsetzung**:

1. **Gleichungen mit genau einer Unbekannten "x"** generieren.
2. **Steigender Schwierigkeitsgrad**:

   * **Level 1: Einfache lineare Gleichungen**

     * z. B. `3x - 7 = 12`, `5 - 2x = 17`, `23 = 7 + 2x`
   * **Level 2: x auf beiden Seiten**

     * z. B. `26 - 4x = 5 + 2x`
   * **Level 3: Mit einer Klammer**:

     * **+ vor der Klammer**: `4 + (2x - 4) = 4x + 7`
     * **- vor der Klammer**: `3x - 3 = 12 - (x - 7)`
     * **× vor der Klammer (implizit!)**: `13(2x - 7) = 12 - 4x`
   * **Level 4: Kombination mehrerer Klammerarten**

     * z. B. `13(2x - 7) = 12 - (x - 7) + (2x - 4)`
   * **Level 5: Brüche**

     * Von `½x = 5` bis zu:

       ```
       (3(4x - 2))/(-2(3x - 12)) = ((7x - 2) * 3)/((-6)(2x - 3))
       ```

---

### 🧠 **Mathematische Präzisierung**:

Du willst eine Klasse linearer Gleichungen der Form:

#### **Allgemein:**

```
A(x) = B(x)
```

wo `A(x)` und `B(x)` Terme sind, die sich aus:

* **konstanten Zahlen**
* **einfachen und mehrfachen x-Terms**
* **Klammerausdrücken**
* **Bruchtermen**
  zusammensetzen.

---

### 📌 **Ergebnisseigenschaft:**

* Die Gleichungen **müssen eindeutig lösbar sein** (eine Lösung).
* Die **Lösung soll häufig ein Bruch** sein, **nicht rund** (z. B. `x = 5/3` statt `x = 2`)
* Brüche sollen auch schwer zu „erraten“ sein – also **optisch unübersichtlich, aber rechnerisch gut lösbar**.

---

### 🛠️ Umsetzung in Python:

Das Skript soll:

1. **Gleichungen generieren** – geordnet nach Schwierigkeitsgrad
2. Die **Lösungen mit Rechenweg** berechnen
3. Beides in **zwei Word-Dateien (`.docx`)** exportieren:

   * Aufgabenblatt (nur Gleichungen, mit Platz zum Rechnen)
   * Lösungsblatt (mit Lösung + Schritt-für-Schritt Rechenweg)

---

### ✅ Output:

* Du bekommst:

  * `arbeitsblatt.docx` mit z. B. 20 Aufgaben (auf Wunsch einstellbar)
  * `loesungsblatt.docx` mit allen Lösungen + Rechenweg

---
