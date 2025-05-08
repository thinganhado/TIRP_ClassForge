/* ─────────────────────  globals  ───────────────────── */
let allStudents  = [];
let filteredRows = [];
let currentPage  = 1;
const rowsPerPage = 30;
const $ = sel => document.querySelector(sel);

/* ─────────────────────  initial load  ───────────────────── */
document.addEventListener("DOMContentLoaded", async () => {
  if (!location.pathname.includes("students")) return;

  await buildClassFilter();          // populate <select>
  await fetchStudents();             // fills allStudents
  buildPager();                      // pager skeleton
  applyFilters();                    // renders first page
  bindUI();
});

/* --- build <select id="class-filter"> ----------------------- */
async function buildClassFilter() {
    const sel = $("#class-filter");
    if (!sel) return;
  
    const list = await fetch("/api/classes").then(r => r.json());   // ["7A", …]
  
    // clear old options except the first (“All Classes”)
    sel.querySelectorAll("option:not(:first-child)").forEach(o => o.remove());
  
    list.forEach(c => {
      const cls = String(c);                    // always a string
      sel.insertAdjacentHTML(
        "beforeend",
        `<option value="${cls.toLowerCase()}">${cls}</option>`
      );
    });
  }  
  
async function fetchStudents(){
  allStudents = await fetch("/api/students").then(r=>r.json());
  filteredRows = [...allStudents];
}

/* ─────────────────────  filters / search  ───────────────────── */
function applyFilters() {
    const txt   = $("#search-input").value.trim().toLowerCase();
    const cls   = $("#class-filter").value;       // already lower-case
  
    filteredRows = allStudents.filter(s => {
      const nameOK  = (`${s.first_name} ${s.last_name}`.toLowerCase()).includes(txt);
      const classOK = !cls || String(s.class).toLowerCase() === cls;
      return nameOK && classOK;
    });
  
    currentPage = 1;
    renderTable();
  }  

/* ─────────────────────  table renderer  ───────────────────── */
function currentPageRows(){
  const start = (currentPage-1)*rowsPerPage;
  return filteredRows.slice(start,start+rowsPerPage);
}
function renderTable(){
  const tbody = $("#students-table-body");
  tbody.innerHTML = currentPageRows().map(s=>`
     <tr data-id="${s.id}">
       <td><input type="checkbox" class="row-check"></td>
       <td>${s.id}</td><td>${s.first_name}</td><td>${s.last_name}</td>
       <td>${s.email}</td><td>${s.house}</td><td>${s.class}</td>
     </tr>`).join("");
  wireRowClicks();
  updateCounters();
  togglePager();
}
function updateCounters(){
  $("#student-count").textContent = `Total Students: ${allStudents.length}`;

  const first = filteredRows.length? (currentPage-1)*rowsPerPage+1 : 0;
  const last  = Math.min(currentPage*rowsPerPage, filteredRows.length);
  $("#student-range").textContent =
        `Showing ${first}–${last} of ${filteredRows.length} students`;
}

/* ─────────────────────  row selection & detail panel  ───────────────────── */

/* click-handler wired after table rows are built */
/* ─────────────────────────────────────────────────────────────
   Student-detail panel
────────────────────────────────────────────────────────────── */
async function showStudentDetails(studentId) {
    const data = await fetch(`/api/students/${studentId}`).then(r => r.json());
  
    const tbl = $("#detail-table");
    const body = tbl.querySelector("tbody");
    body.innerHTML = "";               // wipe old rows
  
    // order & labels we want to show
    const labels = {
      perc_effort   : "Effort %",        // rename if wanted
      attendance    : "Attendance %",
      perc_academic : "Academic %",
      complete_years: "Years Completed",
      status        : "Status"
    };
  
    Object.entries(labels).forEach(([key, label]) => {
      if (key in data && data[key] != null) {
        body.insertAdjacentHTML(
          "beforeend",
          `<tr><td><strong>${label}</strong><br>${data[key]}</td></tr>`
        );
      }
    });
  
    $("#detail-placeholder").style.display = "none";
    tbl.style.display = "";
  }

  function wireRowClicks() {
    $("#students-table-body").querySelectorAll("tr").forEach(row => {
  
      /* clicking anywhere in the row */
      row.onclick = () => selectRow(row);
  
      /* …or clicking the checkbox does the same */
      row.querySelector(".row-check").onchange = e => {
        if (e.target.checked) selectRow(row);
        else {                         // user unticked the only selected row
          row.classList.remove("selected");
          $("#detail-table").style.display = "none";
          $("#detail-placeholder").style.display = "";
        }
      };
    });
  }  

  async function selectRow(row) {

    /* 1. un-tick every checkbox and clear highlight */
    document.querySelectorAll(".row-check").forEach(cb => cb.checked = false);
    document.querySelectorAll("#students-table-body tr.selected")
            .forEach(r => r.classList.remove("selected"));
  
    /* 2. mark THIS row */
    row.classList.add("selected");
    row.querySelector(".row-check").checked = true;
  
    /* 3. fetch & display details */
    const sid  = row.dataset.id;
    const data = await fetch(`/api/students/${sid}`).then(r => r.json());
  
    const labels = {
      perc_effort   : "Effort %",
      attendance    : "Attendance %",
      perc_academic : "Academic %",
      complete_years: "Years Completed",
      status        : "Status"
    };
  
    const body = $("#detail-table tbody");
    body.innerHTML = "";
  
    Object.entries(labels).forEach(([k, label]) => {
      if (k in data && data[k] != null) {
        body.insertAdjacentHTML(
          "beforeend",
          `<tr><td><strong>${label}</strong>${data[k]}</td></tr>`
        );
      }
    });
  
    $("#detail-placeholder").style.display = "none";
    $("#detail-table").style.display       = "";
  }
  
const buildDetailRow = (label,val)=>
  `<tr><td><strong>${label}</strong><br>${val ?? "—"}</td></tr>`;

/* ─────────────────────  pager  ───────────────────── */
function buildPager(){
  $(".table-container").insertAdjacentHTML("beforeend",`
    <div id="pager">
       <button id="prevBtn" disabled>
          <i class="fas fa-angle-left"></i> Prev
       </button>
       <span id="pageInfo">1 / 1</span>
       <button id="nextBtn" disabled>
          Next <i class="fas fa-angle-right"></i>
       </button>
    </div>`);
  $("#prevBtn").onclick = ()=>{currentPage--; renderTable();};
  $("#nextBtn").onclick = ()=>{currentPage++; renderTable();};
}
function togglePager(){
  const pages = Math.max(1, Math.ceil(filteredRows.length/rowsPerPage));
  $("#prevBtn").disabled = currentPage<=1;
  $("#nextBtn").disabled = currentPage>=pages;
  $("#pageInfo").textContent = `${currentPage} / ${pages}`;
}

/* ─────────────────────  misc UI  ───────────────────── */
function bindUI(){
  $("#search-input") .addEventListener("keyup",  applyFilters);
  $("#class-filter").addEventListener("change", applyFilters);
  $("#select-all")  ?.addEventListener("change", e=>{
       document.querySelectorAll(".row-check")
               .forEach(cb=>cb.checked = e.target.checked);
  });

  // expose actions for inline onClick (Print / Export / Create)
  Object.assign(window,{
     printTable : ()=>window.print(),
     exportTable: ()=>alert("Export coming soon"),
     createStudent: ()=>alert("Create student form")
  });
}