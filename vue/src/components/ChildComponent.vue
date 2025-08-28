<template>
  <h1>이름: {{ props.name }}, 성별: {{ props.sex }}</h1>
  <h2>자산: {{ props.assets.join() }}</h2>
  <label>용돈: <input v-model="money" /></label>
  <button @click="sendMoney">전달</button>
  <h2>부모님의 잔소리: {{ message }}</h2>
</template>

<script setup>
import { ref } from "vue";

const props = defineProps({
  name: String,
  sex: String,
  assets: Array,
});

const emit = defineEmits(["send"]);

const money = ref(0);

function sendMoney() {
  const data = {
    name: props.name,
    money: Number(money.value),
  };
  emit("send", data);
}

const message = ref("");

function setMessage(msg) {
  message.value = msg;
}

defineExpose({
  setMessage,
});
</script>
