function loadPage(pageName) {
    document.getElementById("main-title").textContent = pageName;
    document.getElementById("main-content").textContent = `You are now viewing the ${pageName} page.`;
}

function loadPage(pageName) {
    const title = document.getElementById("main-title");
    const content = document.querySelector(".content");

    if (pageName === 'Students') {
        title.textContent = "";
        content.innerHTML = `
            <h1 class="page-title">Students</h1>

            <div class="table-actions">
                <button class="action-button" onclick="printTable()"><i class="fas fa-print"></i> Print</button>
                <button class="action-button" onclick="exportTable()"><i class="fas fa-file-export"></i> Export</button>
                <button class="action-button" onclick="createStudent()"><i class="fas fa-user-plus"></i> Create Student</button>
            </div>

            <div class="table-container">
                <table class="students-table">
                    <thead>
                        <tr>
                            <th><input type="checkbox" id="select-all"></th>
                            <th>ID</th>
                            <th>First Name</th>
                            <th>Last Name</th>
                            <th>Gender</th>
                            <th>Class</th>
                        </tr>
                    </thead>
                    <tbody id="students-table-body"></tbody>
                </table>
            </div>
        `;

        const students = [
            { id: 1, first_name: "Alice", last_name: "Johnson", gender: "Female", class: "7A" },
            { id: 2, first_name: "Liam", last_name: "Smith", gender: "Male", class: "7B" }
        ];

        allStudents = students;

        const tbody = document.getElementById("students-table-body");
        tbody.innerHTML = students.map(s => `
            <tr>
                <td><input type="checkbox" class="row-checkbox"></td>
                <td>${s.id}</td>
                <td>${s.first_name}</td>
                <td>${s.last_name}</td>
                <td>${s.gender}</td>
                <td>${s.class}</td>
            </tr>
        `).join('');
        filterTable();

        const selectAll = document.getElementById("select-all");
        const checkboxes = document.querySelectorAll(".row-checkbox");

        if (selectAll) {
            selectAll.addEventListener("change", function () {
                checkboxes.forEach(cb => cb.checked = this.checked);
            });
        }
    }
}

function printTable() {
    window.print();
}

function exportTable() {
    alert("Export functionality coming soon!");
}

function createStudent() {
    alert("Open student creation form here.");
}

function filterTable() {
    const search = document.getElementById("search-input").value.toLowerCase();
    const classFilter = document.getElementById("class-filter").value.toLowerCase();
    const rows = document.querySelectorAll("#students-table-body tr");

    let visible = 0;

    rows.forEach(row => {
        const cells = row.querySelectorAll("td");
        const fullName = (cells[2].textContent + " " + cells[3].textContent).toLowerCase();
        const classValue = cells[5].textContent.toLowerCase();

        const matchesSearch = fullName.includes(search);
        const matchesClass = classFilter === "" || classValue === classFilter;

        if (matchesSearch && matchesClass) {
            row.style.display = "";
            visible++;
        } else {
            row.style.display = "none";
        }
    });

    const total = allStudents.length; //  now reflects full dataset
    document.getElementById("student-count").textContent = `Total Students: ${total}`;
    document.getElementById("student-range").textContent = `Showing 1–${visible} of ${total} students`;
}
