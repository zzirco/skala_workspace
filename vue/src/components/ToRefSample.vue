<script setup>
import { reactive, ref, toRef, toRefs, unref } from "vue";

const state = reactive({
  name: "홍길동",
  age: 20,
});

const { name, age } = toRefs(state);

const count = ref(20);
const num = 30;

console.log(count, num, unref(count));

// const name = toRef(state, "name");
// const age = toRef(state, "age");
</script>
<template>
  <input v-model="name" />
  <input v-model="age" type="number" />
  <div>{{ state.name }}, {{ state.age }}</div>
</template>
