<script setup>
import { ref, reactive } from 'vue';

import PlayerStocks from "./components/PlayerStocks.vue";
import StockList from "./components/StockList.vue";

</script>

<template>
  <div class="container-fluid">
    <div class="row bg-body-tertiary">
      <div class="col">
        <span class="ps-2 fs-2">SKALA STOCK Market</span>
      </div>
    </div>
    <div class="row">
      <div class="col border m-1">
        <StockList />
      </div>
      <div class="col border m-1">
        <PlayerStocks />
      </div>
    </div>
  </div>
</template>
