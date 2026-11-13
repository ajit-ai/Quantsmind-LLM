const path = require("path");
const Mocha = require("mocha");

function run() {
  return new Promise((resolve, reject) => {
    const mocha = new Mocha({ ui: "bdd", timeout: 90000 });
    mocha.addFile(path.resolve(__dirname, "extension.test.js"));

    mocha.run((failures) => {
      if (failures > 0) {
        reject(new Error(`${failures} VS Code host test(s) failed.`));
      } else {
        resolve();
      }
    });
  });
}

module.exports = { run };
