<template>
  <h1>mouse move</h1>

  <div
    @mousemove="showPosition"
    @mouseleave="clearPosition"
    style="
      width: 400px;
      height: 400px;
      border: 1px solid;
      background-color: grey;
    "
  ></div>
  <p>위치: {{ positionX }} {{ positionY }}</p>
</template>
<script setup>
import { ref } from "vue";

const positionX = ref(0);
const positionY = ref(0);

function showPosition(event) {
  console.log(event);
  positionX.value = event.clientX;
  positionY.value = event.clientY;
}
function clearPosition(event) {
  positionX.value = 0;
  positionY.value = 0;
}
</script>
