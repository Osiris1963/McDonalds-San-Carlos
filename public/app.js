
function generateForecast() {
  const customers = parseInt(document.getElementById('inputCustomers').value) || 0;
  const weather = document.getElementById('inputWeather').value;
  const addons = parseInt(document.getElementById('inputAddon').value) || 0;

  let weatherFactor = 1;
  if (weather === 'rainy') weatherFactor = 0.9;
  if (weather === 'sunny') weatherFactor = 1.1;

  let forecastSales = (customers * 100) * weatherFactor + addons;
  let atv = customers > 0 ? forecastSales / customers : 0;

  document.getElementById('resultOutput').innerHTML = `
    <p>Forecasted Sales: <strong>₱${forecastSales.toFixed(2)}</strong></p>
    <p>Average Transaction Value (ATV): <strong>₱${atv.toFixed(2)}</strong></p>
    <p>Customers: <strong>${customers}</strong></p>
  `;
}
