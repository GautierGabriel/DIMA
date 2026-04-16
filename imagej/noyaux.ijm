// Macro annotation ImageJ - Comptage de noyaux

// Copier-coller la macro dans Plugins/New/Macro
// Dans Analyze/Set Measurements : cocher Display Label (désactiver tout le reste)
// Modifier le dossier contenant les images (variable dir)
// Cliquer sur Run : la séquence d’images se lance, on peut annoter les images une par une
// Placer les points. ok pour pacer a l'image suivante. accepter la remise a 0 de la selction dans le pop up a la fin de chaque image. 

// Ne jamais fermer la fenêtre des résultats, il faut la sauvegarder manuellement à la fin (fichier .csv)


// Réinitialisation de l'interface
output = "Results";
if (isOpen("Results")) {
    selectWindow("Results");
    run("Close");
}
run("Clear Results");

// Sélection du dossier source
dir = getDirectory("/home/gabriel/ownCloud/travail/Mines/DIMA/Segmentation/imagej/data");

// Filtrage des fichiers PNG
allFiles = getFileList(dir);
names = newArray();
for (i = 0; i < allFiles.length; i++) {
    if (endsWith(allFiles[i], ".png")) {
        names = Array.concat(names, newArray(allFiles[i]));
    }
}

// Chargement de la séquence d'images
run("Image Sequence...", "open=[" + dir + "] sort");

// AGRANDIR L'AFFICHAGE (ex: 300% pour du 512x512)
run("Set... ", "zoom=300");

n = nSlices;

// Boucle de traitement par image
for (i = 1; i <= n; i++) {
    setSlice(i);
    name = names[i-1];
    
    // Activation de l'outil Multi-point
    setTool("multi-point");

    // Instruction à l'utilisateur
    // L'utilisateur place autant de points que nécessaire
    waitForUser("Image "+ i + "/" + n + ": " + name + "\n" +
                "1. Marquez chaque noyau d'un point.\n" + 
                "2. Cliquez sur OK une fois terminé.");

    // Récupération des coordonnées de tous les points placés
    getSelectionCoordinates(xpoints, ypoints);
    numPoints = xpoints.length;

    // Enregistrement des données dans la table des résultats
    if (numPoints > 0) {
        for (j = 0; j < numPoints; j++) {
            row = nResults;
            setResult("Image", row, name);
            setResult("Slice_Index", row, i);
            setResult("Nucleus_ID", row, j + 1);
            setResult("X", row, xpoints[j]);
            setResult("Y", row, ypoints[j]);
        }
        updateResults();
    }

    // Effacer la sélection avant de passer à l'image suivante
    run("Select None");
}

// Message de fin
showMessage("Annotation terminée", "Toutes les images ont été traitées.\n" +
            "N'oubliez pas de sauvegarder la table des résultats au format .csv");
