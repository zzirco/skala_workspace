<template>
    <label>
        색상<input v-model="inputColor">
    </label>
    <label>
        사이즈<input  v-model.number="inputSize" type="text"></input>
    </label>
    <button @click="applystyle">적용</button>
    <div :style="styleName">
        style shift
    </div>

</template>
<script setup>
import { ref } from "vue"
const inputColor = ref('')

const inputSize = ref('')
const styleName = ref({});
function applystyle() {
    console.log(typeof inputSize.value)
    styleName.value = {
        width: inputSize.value + "px",
        height: inputSize.value + "px",
        backgroundColor: inputColor.value,
        border: "1px solid"
    }
}
</script>
<style>
.base {
  padding: 10px 16px;
  font-size: 16px;
  margin: 8px;
  width: 200px;
  height: 200px;
  border: #42B983;
}
.primary {
  background-color: #42B983;
  color: white;
}
.secondary {
  background-color: #ddd;
  color: black;
}
.rounded {
  border-radius: 100px;
  color: #42B983;
}
.highlight {
  box-shadow: 0 0 6px rgba(255, 215, 0, 0.6);
}
</style>