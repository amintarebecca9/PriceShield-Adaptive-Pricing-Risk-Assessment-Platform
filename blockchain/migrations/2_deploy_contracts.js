const MarketOptimizer = artifacts.require("MarketOptimizer");

module.exports = function(deployer) {
  deployer.deploy(MarketOptimizer, { gas: 6000000 });
};
