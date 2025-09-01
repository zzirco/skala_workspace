<template>
  <div class="container min-vh-100 d-flex align-items-center">
    <div class="row w-100 justify-content-center">
      <div class="col-12 col-sm-10 col-md-8 col-lg-6">
        <div class="card shadow-sm">
          <div class="card-body p-4">
            <h1 class="h4 mb-4 text-center">회원가입</h1>

            <form @submit.prevent="handleSubmit" novalidate>
              <!-- 이름 -->
              <div class="mb-3">
                <label for="name" class="form-label">이름</label>
                <input
                  id="name"
                  type="text"
                  class="form-control"
                  v-model.trim="name"
                  :class="{ 'is-invalid': submitted && !isNameValid }"
                  placeholder="홍길동"
                  required
                />
                <div class="invalid-feedback">이름을 2자 이상 입력하세요.</div>
              </div>

              <!-- 성별 -->
              <div class="mb-3">
                <label class="form-label d-block">성별</label>
                <div
                  class="form-check form-check-inline"
                  v-for="opt in genderOptions"
                  :key="opt.value"
                >
                  <input
                    class="form-check-input"
                    type="radio"
                    :id="`gender-${opt.value}`"
                    name="gender"
                    :value="opt.value"
                    v-model="gender"
                    :class="{ 'is-invalid': submitted && !isGenderValid }"
                  />
                  <label
                    class="form-check-label"
                    :for="`gender-${opt.value}`"
                    >{{ opt.label }}</label
                  >
                </div>
                <div
                  class="invalid-feedback d-block"
                  v-if="submitted && !isGenderValid"
                >
                  성별을 선택하세요.
                </div>
              </div>

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
                <div class="invalid-feedback">
                  올바른 이메일 주소를 입력하세요.
                </div>
              </div>

              <!-- 비밀번호 -->
              <div class="mb-3">
                <label for="password" class="form-label">비밀번호</label>
                <div class="input-group">
                  <input
                    :type="showPw ? 'text' : 'password'"
                    id="password"
                    class="form-control"
                    v-model="password"
                    :class="{ 'is-invalid': submitted && !isPasswordValid }"
                    placeholder="비밀번호 (6자 이상)"
                    minlength="6"
                    required
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

              <!-- 비밀번호 확인 -->
              <div class="mb-1">
                <label for="password2" class="form-label">비밀번호 확인</label>
                <div class="input-group">
                  <input
                    :type="showPw2 ? 'text' : 'password'"
                    id="password2"
                    class="form-control"
                    v-model="password2"
                    :class="{ 'is-invalid': submitted && !isPasswordMatch }"
                    placeholder="비밀번호를 다시 입력"
                    minlength="6"
                    required
                  />
                  <button
                    class="btn btn-outline-secondary"
                    type="button"
                    @click="showPw2 = !showPw2"
                    :aria-pressed="showPw2.toString()"
                  >
                    {{ showPw2 ? "숨기기" : "보기" }}
                  </button>
                  <div
                    class="invalid-feedback d-block"
                    v-if="submitted && !isPasswordMatch"
                  >
                    비밀번호가 일치하지 않습니다.
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
                  회원가입
                </button>
                <button
                  class="btn btn-outline-secondary"
                  type="button"
                  @click="goToLogin"
                >
                  로그인 화면으로
                </button>
              </div>
            </form>

            <!-- 서버 에러 -->
            <div v-if="errorMsg" class="alert alert-danger mt-3" role="alert">
              {{ errorMsg }}
            </div>
            <!-- 성공 안내 -->
            <div
              v-if="successMsg"
              class="alert alert-success mt-3"
              role="alert"
            >
              {{ successMsg }}
            </div>
          </div>
        </div>

        <p class="text-center text-muted mt-3 mb-0" style="font-size: 0.9rem">
          데모 페이지입니다. 실제 API 연동은 <code>handleSubmit</code> 내부를
          교체하세요.
        </p>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed } from "vue";
import { useRouter } from "vue-router";

/**
 * 부모/라우터에서 제어하고 싶다면 아래 emit 사용 가능
 * - onSignup(payload: { name: string; gender: 'm'|'f'|'x'; email: string; password: string })
 * - onGotoLogin()
 */
const emit = defineEmits<{
  (
    e: "signup",
    payload: {
      name: string;
      gender: "m" | "f" | "x";
      email: string;
      password: string;
    }
  ): void;
  (e: "goto-login"): void;
}>();

const name = ref("");
const gender = ref<"m" | "f" | "x" | "">("");
const email = ref("");
const password = ref("");
const password2 = ref("");

const showPw = ref(false);
const showPw2 = ref(false);
const submitted = ref(false);
const isSubmitting = ref(false);
const errorMsg = ref("");
const successMsg = ref("");

const genderOptions = [
  { label: "남", value: "m" as const },
  { label: "여", value: "f" as const },
  { label: "선택 안함", value: "x" as const },
];

const isNameValid = computed(() => name.value.trim().length >= 2);
const isGenderValid = computed(() => !!gender.value);
const isEmailValid = computed(() =>
  /^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(email.value)
);
const isPasswordValid = computed(() => password.value.length >= 6);
const isPasswordMatch = computed(() => password.value === password2.value);
const canSubmit = computed(
  () =>
    isNameValid.value &&
    isGenderValid.value &&
    isEmailValid.value &&
    isPasswordValid.value &&
    isPasswordMatch.value
);

async function handleSubmit() {
  submitted.value = true;
  errorMsg.value = "";
  successMsg.value = "";

  if (!canSubmit.value) return;

  try {
    isSubmitting.value = true;
    // TODO: 실제 회원가입 API로 교체
    // 예: await signUpApi({ name: name.value, gender: gender.value, email: email.value, password: password.value })
    await new Promise((r) => setTimeout(r, 700));

    emit("signup", {
      name: name.value,
      gender: gender.value as "m" | "f" | "x",
      email: email.value,
      password: password.value,
    });
    successMsg.value = "회원가입이 완료되었어요. 이제 로그인할 수 있어요!";
    const router = useRouter();
    await router.push("/login");
  } catch (e: any) {
    errorMsg.value = "회원가입에 실패했습니다. 잠시 후 다시 시도해주세요.";
  } finally {
    isSubmitting.value = false;
  }
}

const router = useRouter();

function goToLogin() {
  router.push("/login");
}
</script>

<style scoped>
.card {
  border-radius: 1rem;
}
</style>
