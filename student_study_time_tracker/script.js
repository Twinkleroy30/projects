document.getElementById('studyForm').addEventListener('submit', function(event) {
    event.preventDefault();

    const subject = document.getElementById('subject').value;
    const duration = document.getElementById('duration').value;
    const date = new Date().toLocaleDateString();

    if (subject && duration) {
        addSessionToTable(subject, duration, date);
        saveSessionToDatabase(subject, duration, date);
    }

    document.getElementById('studyForm').reset();
});

function addSessionToTable(subject, duration, date) {
    const table = document.getElementById('sessionsTable').getElementsByTagName('tbody')[0];
    const newRow = table.insertRow(table.rows.length);

    const cell1 = newRow.insertCell(0);
    const cell2 = newRow.insertCell(1);
    const cell3 = newRow.insertCell(2);

    cell1.innerHTML = subject;
    cell2.innerHTML = duration;
    cell3.innerHTML = date;
}

function saveSessionToDatabase(subject, duration, date) {
    const xhr = new XMLHttpRequest();
    xhr.open('POST', 'save_session.php', true);
    xhr.setRequestHeader('Content-Type', 'application/x-www-form-urlencoded');
    xhr.onreadystatechange = function() {
        if (xhr.readyState === 4 && xhr.status === 200) {
            console.log('Session saved to the database.');
        }
    };
    xhr.send(`subject=${subject}&duration=${duration}&date=${date}`);
}
