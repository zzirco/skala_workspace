<template>
  <div>
    <h2>{{ user.name }} -{{ user.address.city }}</h2>
    <button @click="changeName">이름 변경</button>
    <button @click="changeCity">도시 변경</button>
  </div>
</template>
<script setup>
import { userRef, updateName, updateCity } from "../scripts/shallow-ref.js";
const user = userRef;
function changeName() {
  updateName("임꺽정");
}
function changeCity() {
  updateCity("부산");
}
</script>
