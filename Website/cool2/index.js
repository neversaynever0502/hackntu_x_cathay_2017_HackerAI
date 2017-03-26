(function() {

  var customBackground = document.getElementById("customBackground");

  function addDropdownClass(e) {
    var target = e.target;
    if (target) {
      if (target.nodeName.toLowerCase() === "label") {
        target.parentElement.classList.toggle("active");
      }
    }
  }

  function toggleBackgroundClass(e) {
    var target = e.target;
    if (target) {
      var backgroundClass = target.getAttribute("id");
      if (customBackground) {
        customBackground.className = backgroundClass;
      }
    }
  }

  var customDropdown = document.getElementById("customDropdown");
  if (customDropdown) {
    customDropdown.addEventListener("click", addDropdownClass);
  }

  var customInputs = document.getElementsByClassName("custom-input");
  if (customInputs) {
    Array.prototype.forEach.call(customInputs, function(el, i) {
      el.addEventListener("change", toggleBackgroundClass);
    });
  }

})();