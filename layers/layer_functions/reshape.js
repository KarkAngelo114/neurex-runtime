

const feedforward = (input) => {
    return {
        outputs: input,
        z_values: input,
        incrementor_value:0
    }
}



module.exports = {
    feedforward
}