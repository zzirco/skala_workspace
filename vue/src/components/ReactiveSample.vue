<script setup>
import { ref, reactive } from "vue";

const user = reactive({ name: "홍길동", age: 20 });
const fruits = reactive(["사과", "배"]);

function increaseAge() {
  user.age++;
}

function addFruit(fruit) {
  fruits.push(fruit);
}
</script>
<template>
  <div>
    <p>이름: <input v-model="user.name" /></p>
    <p>나이: {{ user.age }} <button @click="increaseAge">한 살 더</button></p>
    <p>과일 목록: {{ fruits.join(", ") }}</p>
    <button @click="addFruit('귤')">귤 추가</button>
  </div>
</template>
