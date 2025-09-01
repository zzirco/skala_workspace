<template>
  <h1>🧙‍♂️둔갑술 선택 메뉴</h1>
  <button @click="handleClick('호랑이')">호랑이로 둔갑</button>
  <button @click="handleClick('도깨비')">도깨비로 둔갑</button>
</template>
<script setup>
import { useRouter } from "vue-router";

const router = useRouter();

function handleClick(target) {
  router.push({ path: "/disguise", query: { target } });
}
</script>
