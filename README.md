Wir benutzen folgendes Datenset für die Modelle: https://github.com/faezetta/VMMRdb

Die Daten müssen in Dekaden sortiert in den unterordnern von data/small eingeordnet werden.
Die Modelle selbst sind in dem models Ordner zu finden. Gespeicherte Modelle sind in "models/saved_models".
Das "Model_training" Notebook enthält den trainings-loop.

Mit dem "Predict_Image" Notebook können die trainierten Modelle an einzelnen, neuen Bildern getestet werden. Je nach Auswahl müsste der ladepfad verändert werden, um andere Modelle zu laden (vit_linear vs vit_linear_swag). Es liest standardmäßig aus dem "input" Ordner, in dem bereits ein paar Beispielbilder vorhanden sind.
