<template>
  <div class="m-4">
    <h3>Tab One</h3>
    <input v-model="text" placeholder="텍스트를 입력하세요." />
    <p>입력값: {{ text }}</p>
  </div>
</template>
<script setup>
import { ref, onMounted, onActivated, onDeactivated } from "vue";
const text = ref("");
onMounted(() => {
  console.log("TabOne Mounted");
});
onActivated(() => {
  console.log("TabOne Activated");
});
onDeactivated(() => {
  console.log("TabOne Deactivated");
});
</script>
