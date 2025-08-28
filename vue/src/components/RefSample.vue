<script setup>
import { ref } from "vue";

const count = ref(0);
const name = ref("홍길동");
const isActive = ref(true);
const items = ref(["사과", "배"]);
const person = ref({ name: "홍길동", age: 20 });

function increaseCount() {
  count.value++;
}

function toggleActive() {
  isActive.value = !isActive.value;
}

function addItem(item) {
  items.value.push(item);
}
</script>

<template>
  <div>
    <p>카운트: {{ count }}</p>
    <button @click="increaseCount">증가</button>
    <p>이름: <input v-model="name" /></p>
    <p>활성 상태: {{ isActive ? "활성" : "비활성" }}</p>
    <button @click="toggleActive">토글</button>
    <p>과일 목록: {{ items.join(", ") }}</p>
    <button @click="addItem('귤')">과일 추가</button>
    <p>인물: {{ person.name }} - {{ person.age }} 세</p>
    <button @click="person.age++">나이 증가</button>
  </div>
</template>
