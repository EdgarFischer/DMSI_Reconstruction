# Train Model

## Config Dateien

Im Ordner configs ist für jede Architekture (derzeit nur UNet) ein eigenes config file, mit dem man diverse Konfigurationen fürs Model training einstellen kann wie z.b. 

- Input / Output des Models
- Datensatz
- Anzahl der Epochen
- ...

## Training

Modelle werden dadurch trainiert, dass man das dazugehörige runfile im run_files Ordner startet. Z.b.

**nohup python3 -u Unet_3D.py > logs/logfile.log 2>&1 &** 

Damit das training funktioniert muss vorher wie oben beschrieben die Config Datei kongifuriert werden.

# Saved models and log files

## Saved models

Trainierte Modelle werden im Ordner saved_models gespeichert. Die Pfadlogik ist dabei hierarchisch aufgebaut:

```
saved_models/
├── Architecture/
    └── Daten Domäne I/
        └── Daten Domäne II/
            └── Daten Domäne II/
                └── Undersampling Strategy/
                    └── Acceleration factor/
                        └── Truncate_t/
                            └── Sampling Mask/
                                └── Datensatz/
                                    └── Model

```
**Erklärung:**

/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Process_Results/7T_DMI/Torso_2H/fn_WB_DMI_bow_250417_glucose/output_44_12mm_hamming

größere Matrix: 

/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Process_Results/7T_DMI/Torso_2H/fn_WB_DMI_bow_250417_glucose/output_56_9mm_hamming
Die Hierarchie wird detailiert nochmal im congif Datei beschrieben, hier eine kurze Zusammenfassung:

- Architecure: Z.B Unet
- Daten Domäne I: Daten sind 5D, Netzwerk 3D, welche der 3 Dimensionen sollen berücksichtigt werden?
- Daten Domäne II: Low Rank Einstellungen
- Undersampling Strategy: Possoing oder Regular?
- Acceleration factor: Selbst erklärend
- Truncate t: Wie viele FID Schritte werden fürs training genutzt?
- Sampling Mask: Nur complementary aktuell
- Datensatz: Verfügbare Datensätze siehe Daten Section
- Model: Hier ist das Modell tatsächlich gespeichert


## Logfiles 

Die logfiles zu den trainierten modellen werden nach der selben Hierarchie Logik wie saved_models oben gespeichert, allerdings im Ordner log_files. Zusätzlich ist im jeden logfile Ordner das Config file reinkopiert mit dem das Model trainiert wurde, für exakte Reproduzierbarkeit.



# Bisherige Datensätze

Neuste Daten von Wolfang:

kleine Matrix: 

/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Process_Results/7T_DMI/Torso_2H/fn_WB_DMI_bow_250417_glucose/output_44_12mm_hamming

- 20 Ringe
- 280 Punkte pro Ring
- 12 Channel
- Radien stimmen nicht mit größerer Matrix überein (also die ersten 20 davon)
- Glucose wurde früher eingenommen (eben vor kleiner Matrix Messung)
- kz positions corrupted (90% 0)!!
- Maske korrupt: Nur 1 Einträge

größere Matrix: 

/ceph/mri.meduniwien.ac.at/departments/radiology/mrsbrain/lab/Process_Results/7T_DMI/Torso_2H/fn_WB_DMI_bow_250417_glucose/output_56_9mm_hamming

- 28 Ringe
- 476 Punkte pro Ring
- 12 Channel
- kz positions corrupted (90% 0)!!
- Maske korrupt: Nur 1 Einträge
