function forecast() {
  const date = document.getElementById("dateInput").value;
  const cust = parseInt(document.getElementById("custInput").value || 0);
  const weather = document.getElementById("weatherInput").value;
  const addon = parseInt(document.getElementById("addonInput").value || 0);

  // Simple forecast logic
  let baseSales = cust * 50;
  if (weather === "Rainy") baseSales *= 0.9;
  if (weather === "Sunny") baseSales *= 1.1;

  const finalSales = baseSales + addon;

  document.getElementById("result").innerHTML = 
    `<h3>Forecast Result</h3>
     <p>Date: ${date}</p>
     <p>Forecasted Sales: ₱${finalSales.toFixed(2)}</p>`;
}
