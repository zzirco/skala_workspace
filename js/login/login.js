// login.js - 이벤트 처리
document.addEventListener("DOMContentLoaded", function () {
  const loginForm = document.getElementById("loginForm");
  const emailInput = document.getElementById("email");
  const passwordInput = document.getElementById("password");
  const emailError = document.getElementById("emailError");
  const passwordError = document.getElementById("passwordError");
  const successMessage = document.getElementById("successMessage");

  // 실시간 검증 (입력 시 에러 메시지 제거)
  emailInput.addEventListener("input", function () {
    if (this.value.trim()) {
      this.classList.remove("error");
      emailError.classList.remove("show");
    }
  });

  passwordInput.addEventListener("input", function () {
    if (this.value.trim()) {
      this.classList.remove("error");
      passwordError.classList.remove("show");
    }
  });

  // 폼 제출 처리
  loginForm.addEventListener("submit", function (e) {
    e.preventDefault();

    const email = emailInput.value;
    const password = passwordInput.value;

    let isValid = true;

    // 이메일 검증
    if (!isNotEmpty(email, "이메일")) {
      emailInput.classList.add("error");
      emailError.classList.add("show");
      isValid = false;
    } else {
      emailInput.classList.remove("error");
      emailError.classList.remove("show");
    }

    // 비밀번호 검증
    if (!isNotEmpty(password, "비밀번호")) {
      passwordInput.classList.add("error");
      passwordError.classList.add("show");
      isValid = false;
    } else {
      passwordInput.classList.remove("error");
      passwordError.classList.remove("show");
    }

    // 모든 검증이 통과된 경우
    if (isValid) {
      successMessage.classList.add("show");

      // 3초 후 google.com으로 이동
      setTimeout(function () {
        window.location.href = "https://google.com";
      }, 1000);
    }
  });
});
