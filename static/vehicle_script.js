function clearForm() {
    document.getElementById("insuranceForm").reset();
}

predictButton.addEventListener('click', function() {
    predictInsurance(); ked
});


function clearFormApp() {
    // Reset the form fields
    document.getElementById("app-id").value = "";
    document.getElementById("app-name").value = "";
}

// Add event listener to the "Clear" button
document.querySelector(".clear-button").addEventListener('click', function() {
    // Call the clearForm function when the "Clear" button is clicked
    clearFormApp();
});