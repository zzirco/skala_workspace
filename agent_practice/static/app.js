const form = document.getElementById("agent-form");
const runBtn = document.getElementById("run-btn");
const stepsEl = document.getElementById("steps");
const globalStatus = document.getElementById("global-status");

const STEP_KEYS = ["send", "clone", "analyze", "generate", "commit", "done"];

function setStepState(key, state) {
  const li = stepsEl.querySelector(`li[data-step="${key}"]`);
  if (!li) return;
  li.classList.remove("pending", "active", "done", "fail");
  li.classList.add(state);
  const label =
    li.querySelector(".label") ||
    (() => {
      const s = document.createElement("span");
      s.className = "label";
      li.appendChild(s);
      return s;
    })();
  const dots =
    li.querySelector(".dots") ||
    (() => {
      const s = document.createElement("span");
      s.className = "dots";
      li.appendChild(s);
      return s;
    })();
  label.textContent =
    li.firstChild?.nodeType === 3
      ? li.firstChild.nodeValue.trim()
      : li.textContent.trim();
  li.childNodes.forEach((n) => {
    if (n.nodeType === 3) n.remove();
  });

  if (state === "active") dots.style.display = "inline";
  else dots.style.display = "none";
}

function resetSteps() {
  STEP_KEYS.forEach((k) => setStepState(k, "pending"));
  globalStatus.classList.remove("ok", "err");
  globalStatus.textContent = "대기 중";
}

function markAllDone() {
  STEP_KEYS.forEach((k) => setStepState(k, "done"));
  globalStatus.classList.add("ok");
  globalStatus.textContent = "완료";
}

function markFail(currentKey, msg) {
  setStepState(currentKey, "fail");
  globalStatus.classList.add("err");
  globalStatus.textContent = `실패: ${msg || ""}`.trim();
}

function optimisticProgressController() {
  let idx = 0;
  let timer = null;

  function start() {
    setStepState(STEP_KEYS[0], "active");
    timer = setInterval(() => {
      if (idx < STEP_KEYS.length - 2) {
        setStepState(STEP_KEYS[idx], "done");
        idx += 1;
        setStepState(STEP_KEYS[idx], "active");
      }
    }, 2500);
  }

  function success() {
    if (timer) clearInterval(timer);
    for (let i = idx; i < STEP_KEYS.length - 1; i++) {
      setStepState(STEP_KEYS[i], "done");
    }
    setStepState("done", "done");
  }

  function fail(msg) {
    if (timer) clearInterval(timer);
    const currentKey = STEP_KEYS[Math.min(idx, STEP_KEYS.length - 2)];
    markFail(currentKey, msg);
  }

  return { start, success, fail };
}

form.addEventListener("submit", async (e) => {
  e.preventDefault();
  runBtn.disabled = true;
  runBtn.textContent = "실행중…";
  resetSteps();

  const fd = new FormData(form);
  const payload = {
    repo_url: fd.get("repo_url"),
    git_commit_name: fd.get("git_commit_name"),
    git_commit_email: fd.get("git_commit_email"),
    github_token: fd.get("github_token"),
    final_filename: fd.get("final_filename"),
    branch_name: fd.get("branch_name") || null,
    open_pr: fd.get("open_pr") === "on",
  };

  const prog = optimisticProgressController();
  prog.start();

  try {
    const res = await fetch("./run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!res.ok) {
      prog.fail(`HTTP ${res.status}`);
      return;
    }

    let ok = false;
    try {
      const txt = await res.text();
      if (txt) {
        const j = JSON.parse(txt);
        ok = !!j.ok;
      }
    } catch (_) {
      ok = true;
    }

    if (ok) {
      prog.success();
      markAllDone();
    } else {
      prog.fail("서버 오류");
    }
  } catch (err) {
    prog.fail(err.message || String(err));
  } finally {
    runBtn.disabled = false;
    runBtn.textContent = "에이전트 실행";
  }
});
