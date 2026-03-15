(function() {
  "use strict";

  const divisionSelect = document.getElementById("division");
  const fighter1Select = document.getElementById("fighter1");
  const fighter2Select = document.getElementById("fighter2");
  const weightClassSelect = document.getElementById("weight-class");
  const btnPredict = document.getElementById("btn-predict");
  const matchupDisplay = document.getElementById("matchup-display");
  const predictionResult = document.getElementById("prediction-result");
  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".panel");

  // Filter fighters by division
  function filterFightersByDivision() {
    const div = divisionSelect.value;
    [fighter1Select, fighter2Select].forEach(sel => {
      const current = sel.value;
      Array.from(sel.options).forEach(opt => {
        if (opt.value === "") {
          opt.style.display = "";
          return;
        }
        const show = opt.dataset.div === div;
        opt.style.display = show ? "" : "none";
      });
      if (fighter1Select.querySelector(`option[value="${current}"]`)?.style.display === "none") {
        fighter1Select.value = "";
      }
      if (fighter2Select.querySelector(`option[value="${current}"]`)?.style.display === "none") {
        fighter2Select.value = "";
      }
    });
  }

  divisionSelect.addEventListener("change", filterFightersByDivision);
  filterFightersByDivision();

  // Tab switching
  tabs.forEach(tab => {
    tab.addEventListener("click", (e) => {
      e.preventDefault();
      const target = tab.dataset.tab;
      tabs.forEach(t => t.classList.remove("active"));
      panels.forEach(p => {
        p.classList.remove("active");
        if ((target === "predict" && p.id === "predict-panel") ||
            (target === "upcoming" && p.id === "upcoming-panel")) {
          p.classList.add("active");
        }
      });
      tab.classList.add("active");
    });
  });

  // Load fighter stats and run prediction
  function updateFighterStats() {
    const f1 = fighter1Select.value;
    const f2 = fighter2Select.value;
    if (!f1 || !f2) {
      matchupDisplay.style.display = "none";
      return;
    }

    const params = new URLSearchParams({ fighter1: f1, fighter2: f2 });
    fetch(`/api/fighter-stats?${params}`)
      .then(r => r.json())
      .then(data => {
        matchupDisplay.style.display = "block";
        predictionResult.style.display = "none";

        document.getElementById("f1-name").textContent = data.fighter1?.name || f1;
        document.getElementById("f1-record").textContent = data.fighter1?.record || "--";
        document.getElementById("f1-height").textContent = data.fighter1?.height || "--";
        document.getElementById("f1-reach").textContent = data.fighter1?.reach || "--";
        document.getElementById("f1-stance").textContent = data.fighter1?.stance || "--";

        document.getElementById("f2-name").textContent = data.fighter2?.name || f2;
        document.getElementById("f2-record").textContent = data.fighter2?.record || "--";
        document.getElementById("f2-height").textContent = data.fighter2?.height || "--";
        document.getElementById("f2-reach").textContent = data.fighter2?.reach || "--";
        document.getElementById("f2-stance").textContent = data.fighter2?.stance || "--";

        document.getElementById("weight-class-label").textContent =
          (weightClassSelect.options[weightClassSelect.selectedIndex]?.text || "Middleweight").toUpperCase();
      })
      .catch(() => {
        matchupDisplay.style.display = "block";
        document.getElementById("f1-name").textContent = f1;
        document.getElementById("f2-name").textContent = f2;
      });
  }

  fighter1Select.addEventListener("change", updateFighterStats);
  fighter2Select.addEventListener("change", updateFighterStats);
  weightClassSelect.addEventListener("change", () => {
    const label = document.getElementById("weight-class-label");
    if (label) label.textContent = (weightClassSelect.options[weightClassSelect.selectedIndex]?.text || "").toUpperCase();
  });

  btnPredict.addEventListener("click", () => {
    const f1 = fighter1Select.value;
    const f2 = fighter2Select.value;
    if (!f1 || !f2 || f1 === f2) {
      alert("Please select two different fighters.");
      return;
    }

    btnPredict.disabled = true;
    fetch("/api/predict", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        fighter1: f1,
        fighter2: f2,
        weight_class: weightClassSelect.value,
        number_of_rounds: document.getElementById("rounds").value,
      }),
    })
      .then(r => r.json().then(j => (r.ok ? j : Promise.reject(j))))
      .then(data => {
        predictionResult.style.display = "block";
        document.getElementById("pred-winner").textContent = data.winner;
        document.getElementById("pred-confidence").textContent = (data.confidence * 100).toFixed(1);
      })
      .catch(err => {
        alert(err?.error || "Prediction failed.");
      })
      .finally(() => {
        btnPredict.disabled = false;
      });
  });

  // Initial stats if both selected on load
  if (fighter1Select.value && fighter2Select.value) {
    updateFighterStats();
  }
})();
