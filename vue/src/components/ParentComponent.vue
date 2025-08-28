<template>
  <h1>자식이 준 용돈: {{ proketMoney }}</h1>
  <ChildComponent
    ref="childRef"
    :name="children[0].name"
    :sex="children[0].sex"
    :assets="children[0].assets"
    @send="addPocktMoney"
  />
  <ChildComponent
    :name="children[1].name"
    :sex="children[1].sex"
    :assets="children[1].assets"
    @send="addPocktMoney"
  />
  <label>메시지: <input v-model="message" /></label>
  <button @click="sendMessage">잔소리</button>
</template>

<script setup>
import { ref } from "vue";
import ChildComponent from "./ChildComponent.vue";

const children = [
  { name: "홍길동", sex: "남자", assets: ["창", "칼"] },
  { name: "홍수아", sex: "여자", assets: ["돈", "집"] },
];

const proketMoney = ref(0);

function addPocktMoney(data) {
  console.log("data", data);
  proketMoney.value += data.money;
}

const message = ref("");

const childRef = ref(null);

console.log("childRef: ", childRef.value);

function sendMessage() {
  console.log("childRef2: ", childRef.value);
  childRef.value?.setMessage(message.value);
}
</script>
