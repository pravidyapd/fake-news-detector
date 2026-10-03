function toggleMode(){
  document.body.classList.toggle("dark");
  try {
    localStorage.setItem("darkMode", document.body.classList.contains("dark") ? "1" : "0");
  } catch (e) {}
}

(function(){
  try {
    if (localStorage.getItem("darkMode") === "1") {
      document.body.classList.add("dark");
    }
  } catch (e) {}
})();