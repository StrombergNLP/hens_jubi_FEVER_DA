
function doAction() {
    let claim = $("#claim").val();
    if (!claim || claim.trim() === "" || !claim.includes(" ")) {
        alert("Manglende input")
    } else {
        hideButton();
        showEvidence();
        getEvidence();
    }
}

function getEvidence() {
    let claim = $("#claim").val();
    $.get("https://evidence-retrieval-dafave.azurewebsites.net/api/evidence-retrieval?claim=" + claim, function(evidence) {
        $("#evidence").html(evidence); 
        var x = document.getElementById("evidenceheader")
        x.style.display = "block"
        getPrediction(claim, evidence);
    });    
}

function getPrediction(claim, evidence) {
    $.get("https://classifier-dafave.azurewebsites.net/api/classifier-function?claim=" + claim + "&evidence=" + evidence, function(prediction) {
        // $("#prediction").html(prediction);
        $(".loader").hide()
        showPrediction(prediction)
    });
}

function hideButton() {
    var x = document.getElementById("gobutton");
    x.style.display = "none"; 
}

function showEvidence() {
    var x = document.getElementById("evidence-output");
    x.style.display = "block";
}

function showPrediction(prediction) {

    if(prediction === "Refuted") {
        var x = document.getElementById("Refuted");
        x.style.backgroundColor = "#933438";
        x.style.color = "#f8f7f8";
    } else if(prediction === "Supported") {
        var x = document.getElementById("Supported");
        x.style.backgroundColor = "#458e57";
        x.style.color = "#f8f7f8";
    } else {
        var x = document.getElementById("NotEnoughInfo");
        x.style.backgroundColor = "#5189a0";
        x.style.color = "#f8f7f8";
    }

    var y = document.getElementById("prediction-output");
    y.style.display = "block";
    var z = document.getElementById("prediction");
    z.style.display = "flex";
}