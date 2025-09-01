<script setup lang="ts">
import { ref } from "vue";
import apiCall from "@/scripts/api-call";
import { storePlayer } from "@/scripts/store-player";
import { useRouter } from "vue-router";

const isNewPlayer = ref(false);
const id = ref("");
const password = ref("");
const router = useRouter();

const login = async () => {
  const response = await apiCall.post("/api/players/login", null, {
    playerId: id.value,
    playerPassword: password.value,
  });

  if (response.result === apiCall.Response.SUCCESS) {
    isNewPlayer.value = false;
    console.log(response);
    storePlayer(response.body);
    router.push("/stock");
  } else {
    isNewPlayer.value = true;
    console.log("등록된 유저 없음");
    console.log(response.body);
  }
};

const signup = async () => {
  const response = await apiCall.post("/api/players", null, {
    playerId: id.value,
    playerPassword: password.value,
  });

  console.log(id.value, password.value);

  if (response.result === apiCall.Response.SUCCESS) {
    isNewPlayer.value = false;
  } else {
    isNewPlayer.value = true;
  }
};
</script>

<template>
  <div class="container-sm mt-3 border border-2 p-1" style="max-width: 600px">
    <div class="bss-background p-1">
      <div class="mt-3 d-flex justify-content-center" style="height: 230px">
        <span class="text-center text-danger fs-1 fw-bold mt-4"
          >SKALA STOCK Market</span
        >
      </div>
      <div class="row bg-info-subtle p-2 m-1" style="opacity: 95%">
        <div class="col">
          <InlineInput
            v-model="id"
            label="플레이어ID"
            class="mb-1"
            type="text"
            placeholder="플레이어ID"
          />
          <InlineInput
            v-model="password"
            label="비밀번호"
            class="mb-1"
            type="password"
            placeholder="비밀번호"
          />
        </div>
        <div class="d-flex justify-content-end">
          <button
            v-if="isNewPlayer"
            class="btn btn-primary btn-sm"
            @click="signup"
          >
            회원가입
          </button>
          <button v-else class="btn btn-primary btn-sm" @click="login">
            로그인
          </button>
        </div>
      </div>
    </div>
  </div>
</template>

<style scoped>
.bss-background {
  width: 590px;
  height: 380px;
  background-image: url("/logo.png");
  background-size: cover;
  background-position: center;
  background-repeat: no-repeat;
}
</style>
