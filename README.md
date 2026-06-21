# Waveguide Opt

Automationsschicht fuer die Waveguide-Optimierung des Bliesma T34B mit Purifi PTT8.

## Schnellstart

```powershell
python -m waveguide_opt target-table
python -m waveguide_opt generate --settings settings.local.json --name start_40mm --output configs/generated/start_40mm.cfg
python -m waveguide_opt batch --count 20 --seed 1
python -m waveguide_opt ui
```

Der ausfuehrliche Ablauf steht in [docs/WORKFLOW.md](docs/WORKFLOW.md).

Die WebApp unter `http://127.0.0.1:8765/` startet schnelle Sampling-Laeufe ohne ATH, zeigt Fortschritt und Log, erlaubt die freie Eingrenzung der Optimierungsparameter und visualisiert die aktuell beste Waveguide-Geometrie als Schnitt- und Frontansicht. Reine WebApp-Laeufe sind auf 5000 Kandidaten begrenzt.

Die Rangfolge nutzt nur noch akustische Metriken aus einem axisymmetrischen Fast-BEM-Modell mit einfacher Finite-Baffle/Wrap-Naeherung. Geometrie wird ausschliesslich ueber die Parametergrenzen eingeschraenkt; es gibt keine Bonuspunkte fuer bestimmte Tiefe, Mundgroesse oder Coverage. Die Grenze `Mund real` bezieht sich auf den tatsaechlich berechneten Munddurchmesser, nicht auf einen losgeloesten Zielparameter. Der akustische Score bewertet konstante Directivity, PTT8-Crossover-Match, Off-axis-Glattheit, On-axis-Ripple und Resonanzindikatoren.

Die Gewichtung dieser akustischen Terme ist in der WebApp einstellbar. Die Ergebnisansicht zeigt fuer die beste Loesung eine Directivity-Map von 0 bis 180 Grad ueber 1-20 kHz mit fester Farbskala von 0 bis -50 dB sowie Beamwidth, Directivity Index und Score-Terme.

Eine spaetere Bempp-cl-Validierung fuer Top-N-Kandidaten ist als separater Schritt vorgesehen. Sie soll dieselben Ergebnisgroessen berechnen und gegen das schnelle Fast-BEM-Modell vergleichen, ohne den normalen Optimierungslauf zu verlangsamen.

ATH/ABEC-Pfade werden in `settings.example.json` dokumentiert. ATH wird erst fuer die spaetere Simulation oder den ABEC-Projektexport ausgewahlter Kandidaten gebraucht. Die ABEC-Integration ist bewusst nicht mehr als aktiver Button in der WebApp verdrahtet, bis der externe ABEC-Ablauf stabil geklaert ist.

## Wichtige Annahmen

- T34B: 34 mm Membran, 7 mm Kalottenhoehe, ca. 40 mm Gesamtstrahldurchmesser inklusive Sicke.
- T34B-Hals: 40 mm Startwert, primaerer Suchbereich 40-41.5 mm.
- PTT8: 173 mm effektiver akustischer Durchmesser aus `Sd = 235.1 cm2`; 221 mm mechanischer Aussendurchmesser nur fuer Layout/Schallwand.
