<script setup>
import { isReactive, isRef, toRaw, unref, ref, reactive } from "vue";

function toJsonString(value) {
  if (isReactive(value)) {
    return JSON.stringify(toRaw(value));
  } else if (isRef(value)) {
    return JSON.stringify(unref(value));
  } else {
    return JSON.stringify(value);
  }
}
const primitive = 123;
const obj = { name: "홍길동", age: 30 };
const arr = [1, 2, 3];
const r1 = ref({ city: "서울" });
const r2 = reactive({ job: "개발자" });

const result = ref([]);
console.log(
  toJsonString(primitive),
  toJsonString(obj),
  toJsonString(arr),
  toJsonString(r1),
  toJsonString(r2)
);
result.value.push(toJsonString(primitive));
result.value.push(toJsonString(obj));
result.value.push(toJsonString(arr));
result.value.push(toJsonString(r1));
result.value.push(toJsonString(r2));
</script>

<template>
  <div>
    {{ result.join() }}
  </div>
</template>
