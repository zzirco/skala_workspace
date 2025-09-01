<template>
  <div class="m-4">
    <h3>Tab Two</h3>
    <p>다른 탭입니다.</p>
  </div>
</template>
<script setup>
import { onMounted, onActivated } from "vue";
onMounted(() => {
  console.log("TabTwo Mounted");
});
onActivated(() => {
  console.log("TabTwo Activated");
});
</script>
