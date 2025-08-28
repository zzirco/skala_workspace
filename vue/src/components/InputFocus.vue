<template>
  <div>
    <button @click="showInputBox">입력창 열기</button>
    <div v-if="isVisible">
      <input ref="inputRef" placeholder="이름 입력" />
    </div>
  </div>
</template>

<script setup>
import { nextTick, ref } from "vue";

const isVisible = ref(false);
const inputRef = ref(null);

async function showInputBox() {
  isVisible.value = true;
  await nextTick();
  setTimeout(() => {
    inputRef.value.focus();
  }, 100);
}
</script>
