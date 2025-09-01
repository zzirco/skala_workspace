<template>
  <div class="container min-vh-100 d-flex align-items-center">
    <div class="row w-100 justify-content-center">
      <div class="col-12 col-sm-10 col-md-8 col-lg-5">
        <div class="card shadow-sm">
          <div class="card-body p-4">
            <h1 class="h4 mb-4 text-center">로그인</h1>

            <form @submit.prevent="handleSubmit" novalidate>
              <!-- 이메일 -->
              <div class="mb-3">
                <label for="email" class="form-label">이메일</label>
                <input
                  id="email"
                  type="email"
                  class="form-control"
                  v-model.trim="email"
                  :class="{ 'is-invalid': submitted && !isEmailValid }"
                  placeholder="name@example.com"
                  required
                />
                <div class="invalid-feedback">올바른 이메일을 입력하세요.</div>
              </div>

              <!-- 비밀번호 -->
              <div class="mb-2">
                <label for="password" class="form-label">비밀번호</label>
                <div class="input-group">
                  <input
                    :type="showPw ? 'text' : 'password'"
                    id="password"
                    class="form-control"
                    v-model="password"
                    :class="{ 'is-invalid': submitted && !isPasswordValid }"
                    placeholder="비밀번호"
                    required
                    minlength="6"
                  />
                  <button
                    class="btn btn-outline-secondary"
                    type="button"
                    @click="showPw = !showPw"
                    :aria-pressed="showPw.toString()"
                  >
                    {{ showPw ? "숨기기" : "보기" }}
                  </button>
                  <div
                    class="invalid-feedback d-block"
                    v-if="submitted && !isPasswordValid"
                  >
                    비밀번호는 최소 6자 이상이어야 합니다.
                  </div>
                </div>
              </div>

              <!-- 액션 버튼 -->
              <div class="d-grid gap-2 mt-4">
                <button
                  class="btn btn-primary"
                  type="submit"
                  :disabled="isSubmitting"
                >
                  <span
                    v-if="isSubmitting"
                    class="spinner-border spinner-border-sm me-2"
                    role="status"
                    aria-hidden="true"
                  ></span>
                  로그인
                </button>
                <button
                  class="btn btn-outline-secondary"
                  type="button"
                  @click="handleSignup"
                >
                  회원가입
                </button>
              </div>
            </form>

            <!-- 에러/알림 -->
            <div v-if="errorMsg" class="alert alert-danger mt-3" role="alert">
              {{ errorMsg }}
            </div>
          </div>
        </div>

        <p class="text-center text-muted mt-3 mb-0" style="font-size: 0.9rem">
          데모 페이지입니다. 실제 API 연동은 handleSubmit 내부를 교체하세요.
        </p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import { useRouter } from "vue-router";

/**
 * 부모에서 잡아쓰고 싶다면 아래 emit을 사용하세요.
 * - onLogin(payload: { email: string; password: string })
 * - onSignup()
 */
const emit = defineEmits<{
  (e: "login", payload: { email: string; password: string }): void;
  (e: "signup"): void;
}>();

const email = ref("");
const password = ref("");
const showPw = ref(false);
const submitted = ref(false);
const isSubmitting = ref(false);
const errorMsg = ref("");

const isEmailValid = computed(() => {
  if (!email.value) return false;
  // 간단한 이메일 검사
  return /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.value);
});
const isPasswordValid = computed(() => password.value.length >= 6);

async function handleSubmit() {
  submitted.value = true;
  errorMsg.value = "";

  if (!isEmailValid.value || !isPasswordValid.value) return;

  try {
    isSubmitting.value = true;
    // TODO: 실제 로그인 API 연동 부분으로 교체
    // 예: await loginApi({ email: email.value, password: password.value })
    // await new Promise((r) => setTimeout(r, 600));

    // emit("login", { email: email.value, password: password.value });
    // 라우터 사용 시:
    // const router = useRouter();
    router.push("/main");
  } catch (e: any) {
    errorMsg.value = "로그인에 실패했습니다. 잠시 후 다시 시도해주세요.";
    console.log(e);
  } finally {
    isSubmitting.value = false;
  }
}

const router = useRouter();

function handleSignup() {
  router.push("/signup");
}
</script>

<style scoped>
/* 작은 디테일 보완 */
.card {
  border-radius: 1rem;
}
</style>
