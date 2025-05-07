// script.js (Updated)

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

            <div class="table-filters">
                <input type="text" id="search-input" placeholder="Search by name..." onkeyup="filterTable()" />
                <select id="class-filter" onchange="filterTable()">
                    <option value="">All Classes</option>
                    <option value="7A">7A</option>
                    <option value="7B">7B</option>
                </select>
            </div>

            <div class="table-summary">
                <span id="student-count">Total Students: 0</span>
                <span id="student-range">Showing 0–0 of 0 students</span>
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

        // Load initial dummy data (simulate future DB query)
        allStudents = [
            { id: 1, first_name: "Alice", last_name: "Johnson", gender: "Female", class: "7A" },
            { id: 2, first_name: "Liam", last_name: "Smith", gender: "Male", class: "7B" }
        ];

        renderStudentTable();
        setupSelectAllCheckbox();
    }
}

function renderStudentTable() {
    const tbody = document.getElementById("students-table-body");
    tbody.innerHTML = allStudents.map(s => `
        <tr>
            <td><input type="checkbox" class="row-checkbox"></td>
            <td>${s.id}</td>
            <td>${s.first_name}</td>
            <td>${s.last_name}</td>
            <td>${s.email}</td>
            <td>${s.house}</td>
            <td>${s.class}</td>
        </tr>
    `).join('');

    filterTable();
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

    document.getElementById("student-count").textContent = `Total Students: ${allStudents.length}`;
    document.getElementById("student-range").textContent = `Showing 1–${visible} of ${allStudents.length} students`;
}

function setupSelectAllCheckbox() {
    const selectAll = document.getElementById("select-all");
    if (selectAll) {
        selectAll.addEventListener("change", function () {
            document.querySelectorAll(".row-checkbox").forEach(cb => cb.checked = this.checked);
        });
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

function toggleSidebar() {
    const sidebar = document.getElementById('sidebar');
    if (sidebar) {
        sidebar.classList.toggle('collapsed');
    }
}

let allStudents = [];

document.addEventListener("DOMContentLoaded", () => {
    if (window.location.pathname.includes("students")) {
        fetch("/api/students")
            .then(res => res.json())
            .then(data => {
                allStudents = data;
                renderStudentTable();
                setupSelectAllCheckbox();
            });
    }
});
