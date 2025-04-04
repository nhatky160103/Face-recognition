<template>
  <div class="hello">
    <d3-network :net-nodes="nodes" :net-links="links" :options="options" />
  </div>
</template>

<script>
import D3Network from "vue-d3-network";

export default {
  name: "Nodes",
  components: {
    D3Network,
  },
  data() {
    return {
      nodes: [],
      links: [],
      nodeSize: 10,
      canvas: false,
    };
  },
  methods: {
    createLinks(characters) {
      let f, t;
      const familyRelationships = [
        { from: "Grandparent 1", to: "Parent 1" },
        { from: "Grandparent 2", to: "Parent 1" },
        { from: "Grandparent 1", to: "Parent 2" },
        { from: "Grandparent 2", to: "Parent 2" },
        { from: "Parent 1", to: "Child 1" },
        { from: "Parent 1", to: "Child 2" },
        { from: "Parent 2", to: "Child 3" },
        { from: "Parent 2", to: "Child 4" },
      ];

      for (const relation of familyRelationships) {
        f = characters.indexOf(relation.from);
        t = characters.indexOf(relation.to);

        if (f !== -1 && t !== -1) {
          this.links.push({ sid: f, tid: t });
        }
      }
    },
    getCount(name) {
      return this.links.filter(link => link.tid === this.nodes.findIndex(node => node.name === name)).length;
    },
  },
  computed: {
    options() {
      return {
        force: 3000,
        size: { w: 800, h: 600 },
        nodeSize: this.nodeSize,
        nodeLabels: true,
        linkLabels: true,
        canvas: this.canvas,
      };
    },
  },

  created() {
    const characters = [
      "Grandparent 1",
      "Grandparent 2",
      "Parent 1",
      "Parent 2",
      "Child 1",
      "Child 2",
      "Child 3",
      "Child 4",
    ];

    this.createLinks(characters);

    for (let j = 0; j < characters.length; j++) {
      this.nodes.push({
        id: j,
        name: characters[j],
        _size: this.getCount(characters[j]) + 20,
        _color: "#" + Math.floor(Math.random() * 16777215).toString(16),
      });
    }
  },
};
</script>

<style>
@import url("https://fonts.googleapis.com/css?family=PT+Sans");
canvas {
  left: 0;
  position: absolute;
  top: 0;
}
.net {
  height: 100%;
  margin: 0;
}
.node {
  transition: fill 0.5s ease;
  fill: #dcfaf3;
}
.node.selected {
  stroke: #caa455;
}
.node.pinned {
  stroke: rgba(106, 37, 185, 0.6);
}
.link {
  stroke: rgba(18, 120, 98, 0.3);
}
.link,
.node {
  stroke-linecap: round;
}
.link:hover,
.node:hover {
  stroke: rgba(250, 197, 65, 0.6);
  stroke-width: 5px;
}
.link.selected {
  stroke: rgba(34, 30, 20, 0.6);
}
.curve {
  fill: none;
}
.link-label,
.node-label {
  fill: black;
}
.link-label {
  transform: translateY(-0.5em);
  text-anchor: middle;
}

body {
  overflow-x: hidden;
}

body,
html {
  margin: 0;
  padding: 0;
}
body {
  background-color: #fff;
  font-family: "PT Sans";
}

#app {
  bottom: 0;
  left: 0;
  max-width: 100%;
  position: absolute;
  top: 0;
  width: 100%;
}

#app {
  -moz-user-select: none;
  -ms-user-select: none;
  -webkit-user-select: none;
  text-align: center;
  user-select: none;
}
</style>
