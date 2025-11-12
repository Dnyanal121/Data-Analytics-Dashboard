// Simple helper to show spinner when forms submit
document.addEventListener("DOMContentLoaded", function(){
  const forms = document.querySelectorAll("form");
  forms.forEach(f => {
    f.addEventListener("submit", function(){
      const btn = f.querySelector("button[type='submit']");
      if(btn){
        btn.disabled = true;
        const old = btn.innerHTML;
        btn.setAttribute("data-old", old);
        btn.innerHTML = "Processing...";
      }
    });
  });
});
